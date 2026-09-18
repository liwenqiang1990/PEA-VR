#!/usr/bin/env python3
"""Train the AMP (aligned multiscale pooling) LongEnough video fingerprint model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_SCRIPTS = Path(__file__).resolve().parent.parent
for _d in (_SCRIPTS / "data", _SCRIPTS / "experiments"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from dataset_paths import LONGENOUGH_CACHE, DATA as DATA_ROOT

from longenough_offset0_multiscale import (
    FEATURE_NAMES,
    IdentityBatchSampler,
    LongEnoughMultiScaleDataset,
    MultiScaleTraffic,
    load_dynamic_multiscale,
    normalize_from_base,
    parse_scales,
    save_normalization,
)
from train_longenough_offset0_multiscale_baseline import (
    augment,
    batch_inputs,
    choose_device,
    predict_episode,
    set_seed,
    supervised_contrastive_loss,
    write_csv,
)


SCENARIOS = (
    "random_mixed_5shot",
    "same_actual_mode_2shot",
    "cross_actual_mode_2shot",
    "balanced_actual_modes_3shot",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=LONGENOUGH_CACHE,
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=DATA_ROOT / "longenough_offset0_rapid60_protocol_v1",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DATA_ROOT / "longenough_offset0_rapid60_baseline_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "longenough_offset0_rapid60_swapcanonical_v1",
    )
    parser.add_argument("--scales-ms", default="100,500,2000")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--steps-per-epoch", type=int, default=40)
    parser.add_argument("--ids-per-batch", type=int, default=10)
    parser.add_argument("--runs-per-id", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--style-dim", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument(
        "--content-input",
        choices=("subtract", "observed"),
        default="subtract",
        help="Feed style-subtracted or raw observed tokens to the identity path.",
    )
    parser.add_argument(
        "--temporal-encoder",
        choices=("transformer", "identity"),
        default="transformer",
        help="Use Transformer token mixing or pool the aligned tokens directly.",
    )
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--ce-weight", type=float, default=0.25)
    parser.add_argument("--swap-weight", type=float, default=0.75)
    parser.add_argument("--content-consistency-weight", type=float, default=0.20)
    parser.add_argument("--time-drop", type=float, default=0.03)
    parser.add_argument("--feature-drop", type=float, default=0.01)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ScaleTokenizer(nn.Module):
    def __init__(self, in_dim: int, d_model: int, tokens: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if d_model % 8 == 0 else 4
        self.tokens = int(tokens)
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, d_model, kernel_size=7, padding=3),
            nn.GroupNorm(groups, d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GroupNorm(groups, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        encoded = self.net(value.transpose(1, 2))
        pooled = F.adaptive_avg_pool1d(encoded, self.tokens)
        return pooled.transpose(1, 2)


class AMPEncoder(nn.Module):
    """Factor a run into aligned content tokens and a delivery-style trajectory."""

    def __init__(
        self,
        scales: tuple[int, ...],
        in_dim: int,
        tokens: int,
        d_model: int,
        style_dim: int,
        embedding_dim: int,
        transformer_layers: int,
        num_base_classes: int,
        dropout: float,
        content_input: str = "subtract",
        temporal_encoder: str = "transformer",
    ) -> None:
        super().__init__()
        if content_input not in {"subtract", "observed"}:
            raise ValueError(f"Unsupported content input: {content_input}")
        if temporal_encoder not in {"transformer", "identity"}:
            raise ValueError(f"Unsupported temporal encoder: {temporal_encoder}")
        self.scales = scales
        self.tokens = int(tokens)
        self.content_input = content_input
        self.temporal_encoder = temporal_encoder
        self.tokenizers = nn.ModuleDict(
            {
                str(scale): ScaleTokenizer(in_dim, d_model, tokens, dropout)
                for scale in scales
            }
        )
        self.observation_fusion = nn.Sequential(
            nn.Linear(len(scales) * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.style_encoder = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, style_dim, kernel_size=5, padding=2),
            nn.Tanh(),
        )
        self.style_to_observation = nn.Linear(style_dim, d_model)
        self.content_seed_norm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.content_encoder = (
            nn.TransformerEncoder(
                layer, num_layers=transformer_layers, norm=nn.LayerNorm(d_model)
            )
            if temporal_encoder == "transformer"
            else nn.Identity()
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model + style_dim, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.pool = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embedding_dim),
        )
        self.classifier = nn.Linear(d_model, num_base_classes)

    def observe(self, inputs: dict[int, torch.Tensor]) -> torch.Tensor:
        aligned = [self.tokenizers[str(scale)](inputs[scale]) for scale in self.scales]
        return self.observation_fusion(torch.cat(aligned, dim=2))

    def decode(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat((content, style), dim=2))

    def forward(self, inputs: dict[int, torch.Tensor]) -> dict[str, torch.Tensor]:
        observed = self.observe(inputs)
        style = self.style_encoder(observed.transpose(1, 2)).transpose(1, 2)
        style_component = self.style_to_observation(style)
        content_seed = self.content_seed_norm(
            observed - style_component if self.content_input == "subtract" else observed
        )
        content = self.content_encoder(content_seed)
        pooled = self.pool(torch.cat((content.mean(dim=1), content.amax(dim=1)), dim=1))
        return {
            "observed": observed,
            "style": style,
            "content": content,
            "embedding": F.normalize(self.projection(pooled), dim=1),
            "identity_logits": self.classifier(pooled),
        }


def cross_condition_donors(labels: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
    """Choose another same-identity run, preferring a different bandwidth trace."""
    donor = torch.empty_like(labels)
    for label in labels.unique(sorted=True):
        positions = torch.nonzero(labels == label, as_tuple=False).flatten()
        if len(positions) < 2:
            raise ValueError("Swap reconstruction requires >=2 runs per identity")
        for offset, target in enumerate(positions):
            candidates = positions[conditions[positions] != conditions[target]]
            if not len(candidates):
                candidates = positions[positions != target]
            donor[target] = candidates[offset % len(candidates)]
    return donor


def content_consistency(content: torch.Tensor, donor: torch.Tensor) -> torch.Tensor:
    target = F.normalize(content.float(), dim=2)
    source = F.normalize(content[donor].float(), dim=2)
    return (1.0 - (target * source).sum(dim=2)).mean()


def load_episodes(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"Empty episode file: {path}")
    return rows


def evaluate_frozen(
    embeddings: np.ndarray,
    videos: np.ndarray,
    episodes: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for episode in episodes:
        support = np.asarray(episode["support_indices"], dtype=np.int64)
        query = np.asarray(episode["query_indices"], dtype=np.int64)
        scores = predict_episode(embeddings, videos, support, query)
        for method, accuracy in scores.items():
            rows.append(
                {
                    "split": episode["split"],
                    "scenario": episode["scenario"],
                    "episode": int(episode["episode"]),
                    "method": method,
                    "accuracy": accuracy,
                }
            )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["split"]), str(row["scenario"]), str(row["method"]))
        groups[key].append(float(row["accuracy"]))
    output: list[dict[str, object]] = []
    for (split, scenario, method), values in sorted(groups.items()):
        array = np.asarray(values, dtype=np.float64)
        output.append(
            {
                "split": split,
                "scenario": scenario,
                "method": method,
                "episodes": len(array),
                "mean_accuracy": float(array.mean()),
                "ci95_half_width": float(
                    1.96 * array.std(ddof=1) / math.sqrt(len(array))
                ),
            }
        )
    return output


def metric(
    summary: list[dict[str, object]], scenario: str, method: str = "support_max"
) -> float:
    match = [
        row
        for row in summary
        if row["scenario"] == scenario and row["method"] == method
    ]
    if len(match) != 1:
        raise ValueError(f"Cannot resolve {scenario}/{method}")
    return float(match[0]["mean_accuracy"])


def selection_score(summary: list[dict[str, object]]) -> float:
    return (
        0.50 * metric(summary, "cross_actual_mode_2shot")
        + 0.25 * metric(summary, "random_mixed_5shot")
        + 0.25 * metric(summary, "same_actual_mode_2shot")
    )


def load_swap(
    path: Path,
    scales: tuple[int, ...],
    device: torch.device,
) -> AMPEncoder:
    """Rebuild a trained AMPEncoder from a checkpoint written by `main`."""
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint["args"]
    model = AMPEncoder(
        scales=scales,
        in_dim=len(FEATURE_NAMES),
        tokens=int(args["tokens"]),
        d_model=int(args["d_model"]),
        style_dim=int(args["style_dim"]),
        embedding_dim=int(args["embedding_dim"]),
        transformer_layers=int(args["transformer_layers"]),
        num_base_classes=len(checkpoint["base_classes"]),
        dropout=float(args["dropout"]),
        content_input=str(args.get("content_input", "subtract")),
        temporal_encoder=str(args.get("temporal_encoder", "transformer")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


@torch.no_grad()
def extract_embeddings(
    model: AMPEncoder,
    data: MultiScaleTraffic,
    indices: np.ndarray,
    scales: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    loader = DataLoader(
        LongEnoughMultiScaleDataset(data, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    result = np.zeros((len(data.video), model.projection[-1].out_features), dtype=np.float32)
    model.eval()
    for batch in loader:
        output = model(batch_inputs(batch, scales, device))
        result[batch["sample_index"].numpy()] = output["embedding"].cpu().numpy()
    return result


@torch.no_grad()
def mechanism_audit(
    model: AMPEncoder,
    data: MultiScaleTraffic,
    split: str,
    scales: tuple[int, ...],
    device: torch.device,
) -> dict[str, float]:
    rng = np.random.RandomState(731 + sum(map(ord, split)))
    chosen: list[int] = []
    for video in np.unique(data.video[data.split == split]):
        pool = np.flatnonzero((data.split == split) & (data.video == video))
        first = pool[data.condition[pool] == 0]
        second = pool[data.condition[pool] == 3]
        if len(first) and len(second):
            chosen.extend([int(rng.choice(first)), int(rng.choice(second))])
    loader = DataLoader(LongEnoughMultiScaleDataset(data, np.asarray(chosen)), batch_size=len(chosen))
    batch = next(iter(loader))
    model.eval()
    output = model(batch_inputs(batch, scales, device))
    content = output["content"]
    style = output["style"]
    observed = output["observed"]
    pairs = torch.arange(len(chosen), device=device).reshape(-1, 2)
    source = pairs[:, 0]
    target = pairs[:, 1]
    matched = model.decode(content[source], style[target])
    shuffled_target = target.roll(1)
    shuffled = model.decode(content[source], style[shuffled_target])
    matched_mse = F.mse_loss(matched, observed[target]).item()
    shuffled_mse = F.mse_loss(shuffled, observed[target]).item()
    return {
        "paired_videos": int(len(pairs)),
        "matched_style_mse": float(matched_mse),
        "shuffled_style_mse": float(shuffled_mse),
        "shuffled_over_matched_ratio": float(shuffled_mse / max(matched_mse, 1e-12)),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_comparison(
    baseline_rows: list[dict[str, str]],
    model_rows: list[dict[str, object]],
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    baseline = {
        (row["split"], row["scenario"], int(row["episode"]), row["method"]): float(row["accuracy"])
        for row in baseline_rows
    }
    episode_rows: list[dict[str, object]] = []
    for row in model_rows:
        key = (str(row["split"]), str(row["scenario"]), int(row["episode"]), str(row["method"]))
        if key not in baseline:
            raise ValueError(f"Baseline lacks frozen episode {key}")
        model_accuracy = float(row["accuracy"])
        episode_rows.append(
            {
                **row,
                "baseline_accuracy": baseline[key],
                "delta": model_accuracy - baseline[key],
            }
        )
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in episode_rows:
        grouped[(str(row["split"]), str(row["scenario"]), str(row["method"]))].append(row)
    rng = np.random.RandomState(seed)
    summary: list[dict[str, object]] = []
    for (split, scenario, method), group in sorted(grouped.items()):
        delta = np.asarray([float(row["delta"]) for row in group])
        draws = rng.randint(0, len(delta), size=(2000, len(delta)))
        bootstrap = delta[draws].mean(axis=1)
        summary.append(
            {
                "split": split,
                "scenario": scenario,
                "method": method,
                "episodes": len(group),
                "baseline_accuracy": float(np.mean([float(row["baseline_accuracy"]) for row in group])),
                "model_accuracy": float(np.mean([float(row["accuracy"]) for row in group])),
                "delta": float(delta.mean()),
                "delta_bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                "delta_bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
            }
        )
    return episode_rows, summary


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.steps_per_epoch = 2
        args.ids_per_batch = 5
        args.runs_per_id = 3
        args.d_model = 32
        args.style_dim = 8
        args.embedding_dim = 32
        args.transformer_layers = 1
        args.eval_every = 1
    torch.set_num_threads(max(1, args.torch_threads))
    set_seed(args.seed)
    device = choose_device(args.device)
    scales = parse_scales(args.scales_ms)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = args.protocol_dir / "run_manifest.csv"
    split_path = args.protocol_dir / "class_split.json"
    episode_manifest_path = args.protocol_dir / "episode_manifest.json"
    class_split = json.loads(split_path.read_text(encoding="utf-8"))
    base_classes = [int(value) for value in class_split["base_classes"]]
    data = normalize_from_base(load_dynamic_multiscale(args.cache_root, manifest, scales))
    save_normalization(args.output_dir / "normalization.json", data)
    base_indices = np.flatnonzero(data.split == "base")
    validation_indices = np.flatnonzero(data.split == "validation")
    test_indices = np.flatnonzero(data.split == "test")
    expected = tuple(40 * len(class_split[f"{role}_classes"]) for role in ("base", "validation", "test"))
    if tuple(map(len, (base_indices, validation_indices, test_indices))) != expected:
        raise ValueError("Rapid protocol run counts are inconsistent")

    label_map = {video: label for label, video in enumerate(base_classes)}
    dataset = LongEnoughMultiScaleDataset(data, base_indices, label_map)
    sampler = IdentityBatchSampler(
        dataset,
        ids_per_batch=args.ids_per_batch,
        runs_per_id=args.runs_per_id,
        steps=args.steps_per_epoch,
        seed=args.seed,
        mode_balanced=False,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers)
    validation_episodes = load_episodes(args.protocol_dir / "validation_episodes.jsonl")
    test_episodes = load_episodes(args.protocol_dir / "test_episodes.jsonl")
    model = AMPEncoder(
        scales=scales,
        in_dim=len(FEATURE_NAMES),
        tokens=args.tokens,
        d_model=args.d_model,
        style_dim=args.style_dim,
        embedding_dim=args.embedding_dim,
        transformer_layers=args.transformer_layers,
        num_base_classes=len(base_classes),
        dropout=args.dropout,
        content_input=args.content_input,
        temporal_encoder=args.temporal_encoder,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * args.steps_per_epoch)
    )

    epoch_rows: list[dict[str, object]] = []
    best_score = -math.inf
    best_epoch = -1
    checkpoint_path = args.output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        totals = defaultdict(float)
        for batch in loader:
            inputs = batch_inputs(batch, scales, device)
            inputs = augment(inputs, args.time_drop, args.feature_drop, args.noise_std)
            labels = batch["label"].to(device)
            conditions = batch["condition"].to(device)
            output = model(inputs)
            donor = cross_condition_donors(labels, conditions)
            reconstruction = model.decode(output["content"][donor], output["style"])
            identity_loss = supervised_contrastive_loss(
                output["embedding"], labels, args.temperature
            )
            ce_loss = F.cross_entropy(output["identity_logits"], labels)
            swap_loss = F.smooth_l1_loss(reconstruction, output["observed"].detach())
            consistency_loss = content_consistency(output["content"], donor)
            loss = (
                identity_loss
                + args.ce_weight * ce_loss
                + args.swap_weight * swap_loss
                + args.content_consistency_weight * consistency_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            for key, value in (
                ("loss", loss),
                ("identity_supcon", identity_loss),
                ("identity_ce", ce_loss),
                ("swap_reconstruction", swap_loss),
                ("content_consistency", consistency_loss),
            ):
                totals[key] += float(value.detach().cpu())

        row: dict[str, object] = {
            "epoch": epoch,
            **{key: value / args.steps_per_epoch for key, value in totals.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "validation_selection_score": "",
            "validation_cross_support_max": "",
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            embeddings = extract_embeddings(
                model, data, validation_indices, scales, device, args.eval_batch_size, args.num_workers
            )
            validation_rows = evaluate_frozen(embeddings, data.video, validation_episodes)
            validation_summary = summarize(validation_rows)
            score = selection_score(validation_summary)
            row["validation_selection_score"] = score
            row["validation_cross_support_max"] = metric(
                validation_summary, "cross_actual_mode_2shot"
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(
                    {
                        "schema_version": 1,
                        "model_state": model.state_dict(),
                        "scales_ms": list(scales),
                        "feature_names": list(FEATURE_NAMES),
                        "base_classes": base_classes,
                        "epoch": epoch,
                        "validation_selection_score": score,
                        "args": {
                            key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()
                        },
                    },
                    checkpoint_path,
                )
            print(
                f"[EPOCH {epoch:03d}] loss={row['loss']:.4f} "
                f"val_joint={score:.4f} cross_max={row['validation_cross_support_max']:.4f}",
                flush=True,
            )
        else:
            print(f"[EPOCH {epoch:03d}] loss={row['loss']:.4f}", flush=True)
        epoch_rows.append(row)

    if best_epoch < 0:
        raise RuntimeError("No checkpoint was selected")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    selected = np.concatenate((validation_indices, test_indices))
    embeddings = extract_embeddings(
        model, data, selected, scales, device, args.eval_batch_size, args.num_workers
    )
    final_rows = evaluate_frozen(embeddings, data.video, validation_episodes + test_episodes)
    final_summary = summarize(final_rows)
    baseline_rows = read_csv(args.baseline_dir / "final_episode_metrics.csv")
    paired_rows, paired_summary = paired_comparison(baseline_rows, final_rows, args.seed + 900_000)
    audit = {
        split: mechanism_audit(model, data, split, scales, device)
        for split in ("validation", "test")
    }

    write_csv(args.output_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(args.output_dir / "final_episode_metrics.csv", final_rows)
    write_csv(args.output_dir / "final_summary.csv", final_summary)
    write_csv(args.output_dir / "paired_episode_comparison.csv", paired_rows)
    write_csv(args.output_dir / "paired_summary.csv", paired_summary)
    test_support = {
        row["scenario"]: row
        for row in paired_summary
        if row["split"] == "test" and row["method"] == "support_max"
    }
    gate = {
        "cross_mode_delta_at_least_1pp": test_support["cross_actual_mode_2shot"]["delta"] >= 0.01,
        "random_drop_at_most_0_5pp": test_support["random_mixed_5shot"]["delta"] >= -0.005,
        "same_mode_drop_at_most_0_5pp": test_support["same_actual_mode_2shot"]["delta"] >= -0.005,
        "style_mechanism_non_degenerate": audit["test"]["shuffled_over_matched_ratio"] > 1.01,
    }
    gate["advance_to_full100"] = bool(all(gate.values()))
    summary = {
        "status": (
            "pipeline smoke only; not a result"
            if args.smoke
            else (
                "completed Full-100 controlled confirmation"
                if len(base_classes) == 50
                else "completed reduced-class exploratory run"
            )
        ),
        "task": "class-disjoint ABR-robust video fingerprinting",
        "model": (
            f"aligned-token content_input={args.content_input} "
            f"temporal_encoder={args.temporal_encoder} "
            f"swap_weight={args.swap_weight:g} consistency_weight={args.content_consistency_weight:g}"
        ),
        "data_policy": {
            "bandwidth_or_qoe_as_model_input": False,
            "qoe_or_global_mode_as_training_target": False,
            "bandwidth_condition_used_only_to_prefer cross-condition same-video swap donors": True,
        },
        "selection": "validation-only 0.50 cross-mode + 0.25 random + 0.25 same-mode Support-Max",
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_score,
        "mechanism_audit": audit,
        "gate": gate,
        "paired_results": paired_summary,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "artifacts": {
            "protocol": str(args.protocol_dir / "protocol.json"),
            "protocol_sha256": sha256_file(args.protocol_dir / "protocol.json"),
            "episode_manifest": str(episode_manifest_path),
            "episode_manifest_sha256": sha256_file(episode_manifest_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "baseline_summary": str(args.baseline_dir / "summary.json"),
            "baseline_summary_sha256": sha256_file(args.baseline_dir / "summary.json"),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

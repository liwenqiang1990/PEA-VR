#!/usr/bin/env python3
"""Train a class-disjoint multiscale LongEnough video-identity baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
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


MODE_PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))


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
        default=DATA_ROOT / "longenough_offset0_protocol_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "longenough_offset0_multiscale_baseline_v1",
    )
    parser.add_argument("--scales-ms", default="100,500,2000")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--ids-per-batch", type=int, default=12)
    parser.add_argument("--runs-per-id", type=int, default=4)
    parser.add_argument("--mode-balanced-sampling", action="store_true")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--ce-weight", type=float, default=0.25)
    parser.add_argument("--time-drop", type=float, default=0.04)
    parser.add_argument("--feature-drop", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument(
        "--selection-objective",
        choices=("random_mean", "abr_joint_support"),
        default="random_mean",
    )
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Force a one-epoch, two-step CPU pipeline check; not a paper result.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScaleBranch(nn.Module):
    def __init__(self, in_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if d_model % 8 == 0 else 4 if d_model % 4 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, d_model, kernel_size=7, padding=3),
            nn.GroupNorm(groups, d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(groups, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(groups, d_model),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2))
        pooled = torch.cat((h.mean(dim=2), h.amax(dim=2)), dim=1)
        return self.output(pooled)


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(
        self,
        scales: tuple[int, ...],
        in_dim: int,
        d_model: int,
        embedding_dim: int,
        num_base_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.scales = scales
        self.branches = nn.ModuleDict(
            {str(scale): ScaleBranch(in_dim, d_model, dropout) for scale in scales}
        )
        self.fusion = nn.Sequential(
            nn.Linear(len(scales) * d_model, 2 * d_model),
            nn.LayerNorm(2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
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

    def forward(
        self, inputs: dict[int, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        branches = [self.branches[str(scale)](inputs[scale]) for scale in self.scales]
        pooled = self.fusion(torch.cat(branches, dim=1))
        embedding = F.normalize(self.projection(pooled), dim=1)
        return embedding, pooled, self.classifier(pooled)


def augment(
    inputs: dict[int, torch.Tensor],
    time_drop: float,
    feature_drop: float,
    noise_std: float,
) -> dict[int, torch.Tensor]:
    output: dict[int, torch.Tensor] = {}
    for scale, value in inputs.items():
        x = value.clone()
        valid = x.abs().sum(dim=2, keepdim=True) > 0
        if time_drop > 0:
            keep = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) >= time_drop
            x = torch.where(keep, x, torch.zeros_like(x))
        if feature_drop > 0:
            keep = torch.rand(x.shape[0], 1, x.shape[2], device=x.device) >= feature_drop
            x = torch.where(keep, x, torch.zeros_like(x))
        if noise_std > 0:
            x = torch.where(valid, x + torch.randn_like(x) * noise_std, x)
        output[scale] = x
    return output


def supervised_contrastive_loss(
    embedding: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    z = F.normalize(embedding.float(), dim=1)
    labels = labels.view(-1, 1)
    same = labels.eq(labels.t())
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = same & ~self_mask
    logits = (z @ z.t()) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    return -(
        (log_prob * positive.float()).sum(dim=1)
        / positive.sum(dim=1).clamp_min(1)
    ).mean()


def batch_inputs(
    batch: dict[str, torch.Tensor], scales: tuple[int, ...], device: torch.device
) -> dict[int, torch.Tensor]:
    return {
        scale: batch[f"x_{scale}"].to(device=device, dtype=torch.float32)
        for scale in scales
    }


@torch.no_grad()
def extract_embeddings(
    model: MultiScaleTemporalEncoder,
    data: MultiScaleTraffic,
    indices: np.ndarray,
    scales: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    dataset = LongEnoughMultiScaleDataset(data, indices)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    model.eval()
    result = np.zeros((len(data.video), model.projection[-1].out_features), dtype=np.float32)
    for batch in loader:
        embedding, _, _ = model(batch_inputs(batch, scales, device))
        positions = batch["sample_index"].numpy().astype(np.int64)
        result[positions] = embedding.cpu().numpy()
    return result


def _choose_runs(
    rng: np.random.RandomState,
    pool: np.ndarray,
    support: int,
    query: int,
) -> tuple[np.ndarray, np.ndarray]:
    chosen = rng.choice(pool, size=support + query, replace=False)
    return chosen[:support], chosen[support:]


def sample_episode(
    rng: np.random.RandomState,
    data: MultiScaleTraffic,
    split: str,
    scenario: str,
    episode: int,
    ways: int = 5,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    selected = np.flatnonzero(data.split == split)
    videos = np.unique(data.video[selected])
    by_video = {video: selected[data.video[selected] == video] for video in videos}
    by_mode = {
        (int(video), mode): pool[data.abr_mode[pool] == mode]
        for video, pool in by_video.items()
        for mode in (0, 1, 2)
    }
    support_parts: list[np.ndarray] = []
    query_parts: list[np.ndarray] = []
    support_mode = query_mode = -1

    if scenario == "random_mixed_5shot":
        eligible = [video for video, pool in by_video.items() if len(pool) >= 10]
        chosen_videos = rng.choice(eligible, size=ways, replace=False)
        for video in chosen_videos:
            support, query = _choose_runs(rng, by_video[int(video)], 5, 5)
            support_parts.append(support)
            query_parts.append(query)
    elif scenario == "same_actual_mode_2shot":
        support_mode = query_mode = episode % 3
        eligible = [
            video for video in videos if len(by_mode[(int(video), support_mode)]) >= 4
        ]
        chosen_videos = rng.choice(eligible, size=ways, replace=False)
        for video in chosen_videos:
            support, query = _choose_runs(
                rng, by_mode[(int(video), support_mode)], 2, 2
            )
            support_parts.append(support)
            query_parts.append(query)
    elif scenario == "cross_actual_mode_2shot":
        support_mode, query_mode = MODE_PAIRS[episode % len(MODE_PAIRS)]
        eligible = [
            video
            for video in videos
            if len(by_mode[(int(video), support_mode)]) >= 2
            and len(by_mode[(int(video), query_mode)]) >= 2
        ]
        chosen_videos = rng.choice(eligible, size=ways, replace=False)
        for video in chosen_videos:
            support_parts.append(
                rng.choice(by_mode[(int(video), support_mode)], size=2, replace=False)
            )
            query_parts.append(
                rng.choice(by_mode[(int(video), query_mode)], size=2, replace=False)
            )
    elif scenario == "balanced_actual_modes_3shot":
        eligible = [
            video
            for video in videos
            if all(len(by_mode[(int(video), mode)]) >= 2 for mode in (0, 1, 2))
        ]
        chosen_videos = rng.choice(eligible, size=ways, replace=False)
        for video in chosen_videos:
            support: list[int] = []
            query: list[int] = []
            for mode in (0, 1, 2):
                pair = rng.choice(by_mode[(int(video), mode)], size=2, replace=False)
                support.append(int(pair[0]))
                query.append(int(pair[1]))
            support_parts.append(np.asarray(support, dtype=np.int64))
            query_parts.append(np.asarray(query, dtype=np.int64))
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    if len(chosen_videos) != ways:
        raise ValueError(f"Insufficient eligible videos for {split}/{scenario}")
    return (
        np.concatenate(support_parts),
        np.concatenate(query_parts),
        support_mode,
        query_mode,
    )


def predict_episode(
    embeddings: np.ndarray,
    videos: np.ndarray,
    support: np.ndarray,
    query: np.ndarray,
) -> dict[str, float]:
    classes = np.unique(videos[support])
    support_z = embeddings[support]
    query_z = embeddings[query]
    prototypes = np.stack(
        [support_z[videos[support] == video].mean(axis=0) for video in classes]
    )
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    mean_prediction = classes[np.argmax(query_z @ prototypes.T, axis=1)]
    support_scores = np.column_stack(
        [
            np.max(query_z @ support_z[videos[support] == video].T, axis=1)
            for video in classes
        ]
    )
    max_prediction = classes[np.argmax(support_scores, axis=1)]
    truth = videos[query]
    return {
        "mean_prototype": float(np.mean(mean_prediction == truth)),
        "support_max": float(np.mean(max_prediction == truth)),
    }


def evaluate_episodes(
    embeddings: np.ndarray,
    data: MultiScaleTraffic,
    split: str,
    episodes: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.RandomState(seed)
    rows: list[dict[str, object]] = []
    scenarios = (
        "random_mixed_5shot",
        "same_actual_mode_2shot",
        "cross_actual_mode_2shot",
        "balanced_actual_modes_3shot",
    )
    for scenario_index, scenario in enumerate(scenarios):
        scenario_rng = np.random.RandomState(rng.randint(0, 2**31 - 1) + scenario_index)
        for episode in range(episodes):
            support, query, support_mode, query_mode = sample_episode(
                scenario_rng, data, split, scenario, episode
            )
            scores = predict_episode(embeddings, data.video, support, query)
            for method, accuracy in scores.items():
                rows.append(
                    {
                        "split": split,
                        "scenario": scenario,
                        "episode": episode,
                        "support_mode": support_mode,
                        "query_mode": query_mode,
                        "method": method,
                        "support_runs": len(support),
                        "query_runs": len(query),
                        "accuracy": accuracy,
                    }
                )
    return rows


def evaluate_frozen_episode_file(
    embeddings: np.ndarray,
    data: MultiScaleTraffic,
    path: Path,
) -> list[dict[str, object]]:
    episodes = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not episodes:
        raise ValueError(f"Empty frozen episode file: {path}")
    rows: list[dict[str, object]] = []
    for episode in episodes:
        support = np.asarray(episode["support_indices"], dtype=np.int64)
        query = np.asarray(episode["query_indices"], dtype=np.int64)
        scores = predict_episode(embeddings, data.video, support, query)
        for method, accuracy in scores.items():
            rows.append(
                {
                    "split": episode["split"],
                    "scenario": episode["scenario"],
                    "episode": int(episode["episode"]),
                    "support_mode": int(episode["support_mode"]),
                    "query_mode": int(episode["query_mode"]),
                    "method": method,
                    "support_runs": len(support),
                    "query_runs": len(query),
                    "accuracy": accuracy,
                }
            )
    return rows


def summarize_eval(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["split"], row["scenario"], row["method"])
        groups[key].append(float(row["accuracy"]))
    summary: list[dict[str, object]] = []
    for (split, scenario, method), values in sorted(groups.items()):
        array = np.asarray(values, dtype=np.float64)
        std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        summary.append(
            {
                "split": split,
                "scenario": scenario,
                "method": method,
                "episodes": len(array),
                "mean_accuracy": float(array.mean()),
                "ci95_half_width": 1.96 * std / math.sqrt(len(array)),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_metric(
    summary: list[dict[str, object]], scenario: str, method: str
) -> float:
    matches = [
        row
        for row in summary
        if row["split"] == "validation"
        and row["scenario"] == scenario
        and row["method"] == method
    ]
    if len(matches) != 1:
        raise ValueError(f"Could not resolve {scenario}/{method}")
    return float(matches[0]["mean_accuracy"])


def validation_score(
    summary: list[dict[str, object]], objective: str = "random_mean"
) -> float:
    if objective == "random_mean":
        return summary_metric(summary, "random_mixed_5shot", "mean_prototype")
    if objective == "abr_joint_support":
        return (
            0.50 * summary_metric(summary, "cross_actual_mode_2shot", "support_max")
            + 0.25 * summary_metric(summary, "random_mixed_5shot", "support_max")
            + 0.25 * summary_metric(summary, "same_actual_mode_2shot", "support_max")
        )
    raise ValueError(f"Unknown validation objective: {objective}")


def main() -> None:
    args = parse_args()
    if args.smoke:
        if args.device == "auto":
            args.device = "cpu"
        args.epochs = 1
        args.steps_per_epoch = 2
        args.ids_per_batch = min(args.ids_per_batch, 6)
        args.runs_per_id = min(args.runs_per_id, 3)
        args.d_model = min(args.d_model, 32)
        args.embedding_dim = min(args.embedding_dim, 32)
        args.eval_episodes = min(args.eval_episodes, 12)
        args.eval_every = 1
    scales = parse_scales(args.scales_ms)
    torch.set_num_threads(max(1, args.torch_threads))
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.protocol_dir / "run_manifest.csv"
    split_path = args.protocol_dir / "class_split.json"
    protocol_path = args.protocol_dir / "protocol.json"
    frozen_validation_path = args.protocol_dir / "validation_episodes.jsonl"
    frozen_test_path = args.protocol_dir / "test_episodes.jsonl"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    base_classes = [int(value) for value in split["base_classes"]]
    validation_classes = [int(value) for value in split["validation_classes"]]
    test_classes = [int(value) for value in split["test_classes"]]
    if set(base_classes) & set(validation_classes) or set(base_classes) & set(test_classes) or set(validation_classes) & set(test_classes):
        raise ValueError("Class-disjoint protocol is not disjoint")

    data = normalize_from_base(
        load_dynamic_multiscale(args.cache_root, manifest_path, scales)
    )
    save_normalization(args.output_dir / "normalization.json", data)
    base_indices = np.flatnonzero(data.split == "base")
    validation_indices = np.flatnonzero(data.split == "validation")
    test_indices = np.flatnonzero(data.split == "test")
    expected_counts = tuple(
        40 * len(classes)
        for classes in (base_classes, validation_classes, test_classes)
    )
    actual_counts = (len(base_indices), len(validation_indices), len(test_indices))
    if actual_counts != expected_counts:
        raise ValueError(
            f"Unexpected run counts for class split: {actual_counts} != {expected_counts}"
        )
    if set(np.unique(data.video[base_indices])) != set(base_classes):
        raise ValueError("Base manifest identities disagree with class split")

    base_label_map = {video: label for label, video in enumerate(base_classes)}
    train_dataset = LongEnoughMultiScaleDataset(data, base_indices, base_label_map)
    sampler = IdentityBatchSampler(
        train_dataset,
        ids_per_batch=args.ids_per_batch,
        runs_per_id=args.runs_per_id,
        steps=args.steps_per_epoch,
        seed=args.seed,
        mode_balanced=args.mode_balanced_sampling,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
    )
    model = MultiScaleTemporalEncoder(
        scales=scales,
        in_dim=len(FEATURE_NAMES),
        d_model=args.d_model,
        embedding_dim=args.embedding_dim,
        num_base_classes=len(base_classes),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * args.steps_per_epoch)
    )

    epoch_rows: list[dict[str, object]] = []
    all_validation_rows: list[dict[str, object]] = []
    best_score = -math.inf
    best_epoch = -1
    best_path = args.output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        loss_sum = supcon_sum = ce_sum = 0.0
        for batch in loader:
            inputs = batch_inputs(batch, scales, device)
            labels = batch["label"].to(device)
            first = augment(inputs, args.time_drop, args.feature_drop, args.noise_std)
            second = augment(inputs, args.time_drop, args.feature_drop, args.noise_std)
            merged = {
                scale: torch.cat((first[scale], second[scale]), dim=0)
                for scale in scales
            }
            merged_labels = torch.cat((labels, labels), dim=0)
            embedding, _, logits = model(merged)
            supcon = supervised_contrastive_loss(
                embedding, merged_labels, args.temperature
            )
            ce = F.cross_entropy(logits, merged_labels)
            loss = supcon + args.ce_weight * ce
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach().cpu())
            supcon_sum += float(supcon.detach().cpu())
            ce_sum += float(ce.detach().cpu())

        epoch_row: dict[str, object] = {
            "epoch": epoch,
            "loss": loss_sum / args.steps_per_epoch,
            "supcon_loss": supcon_sum / args.steps_per_epoch,
            "ce_loss": ce_sum / args.steps_per_epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "validation_random_mixed_mean_accuracy": "",
            "validation_selection_score": "",
            "validation_cross_support_max": "",
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            embeddings = extract_embeddings(
                model,
                data,
                validation_indices,
                scales,
                device,
                args.eval_batch_size,
                args.num_workers,
            )
            if frozen_validation_path.is_file():
                validation_rows = evaluate_frozen_episode_file(
                    embeddings, data, frozen_validation_path
                )
            else:
                validation_rows = evaluate_episodes(
                    embeddings,
                    data,
                    split="validation",
                    episodes=args.eval_episodes,
                    seed=args.seed + 100_000,
                )
            all_validation_rows.extend(
                [{"epoch": epoch, **row} for row in validation_rows]
            )
            validation_summary = summarize_eval(validation_rows)
            random_score = validation_score(validation_summary, "random_mean")
            score = validation_score(validation_summary, args.selection_objective)
            epoch_row["validation_random_mixed_mean_accuracy"] = random_score
            epoch_row["validation_selection_score"] = score
            epoch_row["validation_cross_support_max"] = summary_metric(
                validation_summary, "cross_actual_mode_2shot", "support_max"
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
                        "validation_score": score,
                        "args": {
                            key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()
                        },
                    },
                    best_path,
                )
            print(
                f"[EPOCH {epoch:03d}] loss={epoch_row['loss']:.4f} "
                f"val_select={score:.4f} val_random_mean={random_score:.4f}",
                flush=True,
            )
        else:
            print(f"[EPOCH {epoch:03d}] loss={epoch_row['loss']:.4f}", flush=True)
        epoch_rows.append(epoch_row)

    if best_epoch < 0 or not best_path.is_file():
        raise RuntimeError("Training did not produce a validation-selected checkpoint")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    selected_indices = np.concatenate((validation_indices, test_indices))
    selected_embeddings = extract_embeddings(
        model,
        data,
        selected_indices,
        scales,
        device,
        args.eval_batch_size,
        args.num_workers,
    )
    if frozen_validation_path.is_file() and frozen_test_path.is_file():
        final_rows = evaluate_frozen_episode_file(
            selected_embeddings, data, frozen_validation_path
        ) + evaluate_frozen_episode_file(selected_embeddings, data, frozen_test_path)
        episode_policy = "immutable validation/test episode files"
    else:
        final_rows = evaluate_episodes(
            selected_embeddings,
            data,
            split="validation",
            episodes=args.eval_episodes,
            seed=args.seed + 200_000,
        ) + evaluate_episodes(
            selected_embeddings,
            data,
            split="test",
            episodes=args.eval_episodes,
            seed=args.seed + 300_000,
        )
        episode_policy = "legacy seed-derived episodes"
    final_summary = summarize_eval(final_rows)

    write_csv(args.output_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(args.output_dir / "validation_episode_metrics.csv", all_validation_rows)
    write_csv(args.output_dir / "final_episode_metrics.csv", final_rows)
    write_csv(args.output_dir / "final_summary.csv", final_summary)
    run_summary = {
        "status": "pipeline smoke only; not a paper result" if args.smoke else "completed baseline run",
        "task": "class-disjoint few-shot video identity recognition",
        "device": str(device),
        "torch_version": torch.__version__,
        "scales_ms": list(scales),
        "feature_names": list(FEATURE_NAMES),
        "fixed100_included": False,
        "qoe_or_bandwidth_used_as_input": False,
        "abr_mode_usage": "evaluation stratification" + (
            " and base-only batch sampling" if args.mode_balanced_sampling else " only"
        ),
        "best_epoch": best_epoch,
        "checkpoint_selection": (
            "fixed validation random_mixed_5shot mean-prototype accuracy"
            if args.selection_objective == "random_mean"
            else "fixed validation 0.50 cross + 0.25 random + 0.25 same Support-Max"
        ),
        "episode_policy": episode_policy,
        "best_validation_selection_score": best_score,
        "class_counts": {
            "base": len(base_classes),
            "validation": len(validation_classes),
            "test": len(test_classes),
        },
        "run_counts": {
            "base": len(base_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "artifacts": {
            "protocol": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "run_manifest": str(manifest_path),
            "run_manifest_sha256": sha256_file(manifest_path),
            "class_split": str(split_path),
            "class_split_sha256": sha256_file(split_path),
            "checkpoint": str(best_path),
            "checkpoint_sha256": sha256_file(best_path),
        },
        "final_metrics": final_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

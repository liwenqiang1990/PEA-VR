"""Dynamic multiscale views over the LongEnough offset-0 packet cache."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


FEATURE_NAMES = (
    "down_bytes",
    "up_bytes",
    "down_packets",
    "up_packets",
    "down_mean_packet_length",
)
CONDITION_TO_ID = {"bw1": 0, "bw2": 1, "bw4": 2, "bw8": 3}


@dataclass
class MultiScaleTraffic:
    arrays: dict[int, np.ndarray]
    sample_index: np.ndarray
    video: np.ndarray
    split: np.ndarray
    condition: np.ndarray
    repeat: np.ndarray
    abr_mode: np.ndarray
    run_id: np.ndarray
    normalization: dict[str, object] | None = None


def parse_scales(text: str) -> tuple[int, ...]:
    values = tuple(sorted({int(value.strip()) for value in text.split(",") if value.strip()}))
    if not values or any(value <= 0 for value in values):
        raise ValueError("Temporal scales must be positive milliseconds")
    if 60_000 % values[0] or any(value % values[0] for value in values):
        raise ValueError("All scales must divide 60 s and be integer multiples of the finest scale")
    return values


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4000:
        raise ValueError(f"Expected 4,000 protocol rows, found {len(rows)}")
    indices = [int(row["sample_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("Protocol sample_index must be consecutive and row-aligned")
    return rows


def _aggregate_fine(fine: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return fine
    groups = fine.reshape(fine.shape[0] // factor, factor, fine.shape[1])
    output = np.zeros((groups.shape[0], fine.shape[1]), dtype=np.float32)
    output[:, :4] = groups[:, :, :4].sum(axis=1, dtype=np.float64)
    np.divide(
        output[:, 0],
        output[:, 2],
        out=output[:, 4],
        where=output[:, 2] > 0,
    )
    return output


def load_dynamic_multiscale(
    cache_root: Path,
    manifest_path: Path,
    scales_ms: Sequence[int],
) -> MultiScaleTraffic:
    rows = read_manifest(manifest_path)
    scales = tuple(sorted(int(value) for value in scales_ms))
    finest_ms = scales[0]
    fine_bins = 60_000 // finest_ms
    arrays = {
        scale: np.zeros((len(rows), 60_000 // scale, len(FEATURE_NAMES)), dtype=np.float32)
        for scale in scales
    }
    rows_by_condition: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_condition.setdefault(row["condition"], []).append(row)

    checked_packets = 0
    checked_bytes = 0
    for condition in CONDITION_TO_ID:
        condition_rows = rows_by_condition.get(condition, [])
        if len(condition_rows) != 1000:
            raise ValueError(f"Expected 1,000 rows for {condition}, found {len(condition_rows)}")
        shard = cache_root / f"{condition}.npz"
        print(f"[LOAD] {shard}", flush=True)
        with np.load(shard, allow_pickle=False) as data:
            times_all = data["packet_time_relative_ns"]
            directions_all = data["packet_direction"]
            lengths_all = data["packet_length"]
            offsets = data["run_offsets"]
            for row in condition_rows:
                sample = int(row["sample_index"])
                run_index = int(row["cache_run_index"])
                if str(data["run_id"][run_index]) != row["run_id"]:
                    raise ValueError(f"Run ID mismatch at {condition}/{run_index}")
                if int(data["run_video"][run_index]) != int(row["video"]):
                    raise ValueError(f"Video mismatch at {condition}/{run_index}")
                begin, end = map(int, offsets[run_index : run_index + 2])
                times = times_all[begin:end]
                directions = directions_all[begin:end]
                lengths = lengths_all[begin:end]
                elapsed = times - times[0]
                bins = (elapsed // (finest_ms * 1_000_000)).astype(np.int64, copy=False)
                valid = (bins >= 0) & (bins < fine_bins)
                bins = bins[valid]
                directions = directions[valid]
                lengths = lengths[valid]
                down = directions == 0
                up = directions == 1
                if np.any(~(down | up)):
                    raise ValueError(f"Unknown direction code at {condition}/{run_index}")

                fine = np.zeros((fine_bins, len(FEATURE_NAMES)), dtype=np.float32)
                fine[:, 0] = np.bincount(
                    bins[down], weights=lengths[down], minlength=fine_bins
                )
                fine[:, 1] = np.bincount(
                    bins[up], weights=lengths[up], minlength=fine_bins
                )
                fine[:, 2] = np.bincount(bins[down], minlength=fine_bins)
                fine[:, 3] = np.bincount(bins[up], minlength=fine_bins)
                np.divide(
                    fine[:, 0], fine[:, 2], out=fine[:, 4], where=fine[:, 2] > 0
                )
                for scale in scales:
                    arrays[scale][sample] = _aggregate_fine(fine, scale // finest_ms)

                packet_count = int(len(bins))
                total_bytes = int(lengths.sum(dtype=np.uint64))
                if packet_count != int(row["packet_count"]) or total_bytes != int(row["total_bytes"]):
                    raise ValueError(
                        f"Window aggregate mismatch for {row['run_id']}: "
                        f"packets={packet_count}, bytes={total_bytes}"
                    )
                checked_packets += packet_count
                checked_bytes += total_bytes
        print(f"[BINNED] {condition} runs={len(condition_rows)}", flush=True)

    for scale, values in arrays.items():
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid values in {scale} ms view")
        counts = values[:, :, 2:4].sum(axis=(1, 2), dtype=np.float64)
        byte_totals = values[:, :, 0:2].sum(axis=(1, 2), dtype=np.float64)
        expected_counts = np.asarray([int(row["packet_count"]) for row in rows])
        expected_bytes = np.asarray([int(row["total_bytes"]) for row in rows])
        if not np.array_equal(np.rint(counts).astype(np.int64), expected_counts):
            raise ValueError(f"Packet counts changed in {scale} ms aggregation")
        if not np.array_equal(np.rint(byte_totals).astype(np.int64), expected_bytes):
            raise ValueError(f"Byte totals changed in {scale} ms aggregation")

    print(
        f"[READY] runs={len(rows)} packets={checked_packets:,} bytes={checked_bytes:,} "
        f"scales_ms={list(scales)}",
        flush=True,
    )
    return MultiScaleTraffic(
        arrays=arrays,
        sample_index=np.arange(len(rows), dtype=np.int64),
        video=np.asarray([int(row["video"]) for row in rows], dtype=np.int64),
        split=np.asarray([row["split"] for row in rows]),
        condition=np.asarray([CONDITION_TO_ID[row["condition"]] for row in rows], dtype=np.int64),
        repeat=np.asarray([int(row["repeat"]) for row in rows], dtype=np.int64),
        abr_mode=np.asarray([int(row["abr_mode"]) for row in rows], dtype=np.int64),
        run_id=np.asarray([row["run_id"] for row in rows]),
    )


def normalize_from_base(data: MultiScaleTraffic) -> MultiScaleTraffic:
    base = data.split == "base"
    base_runs = int(base.sum())
    if base_runs < 1:
        raise ValueError("Normalization requires at least one base run")
    stats: dict[str, object] = {
        "fit_split": "base",
        "fit_runs": base_runs,
        "transform": "log1p then base-only nonempty-bin z-score; empty bins restored to zero",
        "feature_names": list(FEATURE_NAMES),
        "scales": {},
    }
    normalized: dict[int, np.ndarray] = {}
    for scale, raw in data.arrays.items():
        valid = raw[:, :, 2:4].sum(axis=2) > 0
        transformed = np.log1p(raw.astype(np.float64))
        base_values = transformed[base][valid[base]]
        mu = base_values.mean(axis=0)
        sigma = base_values.std(axis=0)
        sigma[sigma < 1e-6] = 1.0
        transformed = (transformed - mu[None, None, :]) / sigma[None, None, :]
        transformed[~valid] = 0.0
        normalized[scale] = transformed.astype(np.float32)
        stats["scales"][str(scale)] = {
            "bins": int(raw.shape[1]),
            "mean": mu.tolist(),
            "std": sigma.tolist(),
            "base_nonempty_bins": int(valid[base].sum()),
        }
    data.arrays = normalized
    data.normalization = stats
    return data


class LongEnoughMultiScaleDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        data: MultiScaleTraffic,
        indices: np.ndarray,
        base_label_map: dict[int, int] | None = None,
        mode_labels: np.ndarray | None = None,
    ) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.video = data.video[self.indices]
        if mode_labels is not None and len(mode_labels) != len(data.video):
            raise ValueError("mode_labels must align with the global sample table")
        self.abr_mode = (
            data.abr_mode[self.indices]
            if mode_labels is None
            else np.asarray(mode_labels, dtype=np.int64)[self.indices]
        )
        self.base_label_map = base_label_map or {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_index: int) -> dict[str, torch.Tensor]:
        global_index = int(self.indices[local_index])
        item = {
            f"x_{scale}": torch.from_numpy(values[global_index])
            for scale, values in self.data.arrays.items()
        }
        video = int(self.data.video[global_index])
        item.update(
            {
                "sample_index": torch.tensor(global_index, dtype=torch.long),
                "video": torch.tensor(video, dtype=torch.long),
                "label": torch.tensor(self.base_label_map.get(video, -1), dtype=torch.long),
                "abr_mode": torch.tensor(int(self.abr_mode[local_index]), dtype=torch.long),
                "condition": torch.tensor(int(self.data.condition[global_index]), dtype=torch.long),
            }
        )
        return item


class IdentityBatchSampler(Sampler[list[int]]):
    """Identity-balanced batches, optionally covering distinct ABR modes."""

    def __init__(
        self,
        dataset: LongEnoughMultiScaleDataset,
        ids_per_batch: int,
        runs_per_id: int,
        steps: int,
        seed: int,
        mode_balanced: bool = False,
    ) -> None:
        self.ids_per_batch = int(ids_per_batch)
        self.runs_per_id = int(runs_per_id)
        self.steps = int(steps)
        self.seed = int(seed)
        self.mode_balanced = bool(mode_balanced)
        self.epoch = 0
        self.groups = {
            int(video): np.flatnonzero(dataset.video == video)
            for video in np.unique(dataset.video)
        }
        self.modes = dataset.abr_mode
        if self.ids_per_batch < 2 or self.runs_per_id < 2 or self.steps < 1:
            raise ValueError("Sampler requires >=2 identities, >=2 runs/identity, and >=1 step")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.RandomState(self.seed + 1009 * self.epoch)
        videos = np.asarray(sorted(self.groups), dtype=np.int64)
        for _ in range(self.steps):
            chosen = rng.choice(
                videos, size=min(self.ids_per_batch, len(videos)), replace=False
            )
            batch: list[int] = []
            for video in chosen:
                pool = self.groups[int(video)]
                selected: list[int] = []
                if self.mode_balanced:
                    available_modes = np.unique(self.modes[pool])
                    rng.shuffle(available_modes)
                    for mode in available_modes[: self.runs_per_id]:
                        candidates = pool[self.modes[pool] == mode]
                        selected.append(int(rng.choice(candidates)))
                remaining = self.runs_per_id - len(selected)
                if remaining:
                    candidates = np.setdiff1d(pool, np.asarray(selected), assume_unique=False)
                    if not len(candidates):
                        candidates = pool
                    selected.extend(
                        int(value)
                        for value in rng.choice(
                            candidates, size=remaining, replace=len(candidates) < remaining
                        )
                    )
                batch.extend(selected)
            yield batch


def save_normalization(path: Path, data: MultiScaleTraffic) -> None:
    if data.normalization is None:
        raise ValueError("Data have not been normalized")
    path.write_text(
        json.dumps(data.normalization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

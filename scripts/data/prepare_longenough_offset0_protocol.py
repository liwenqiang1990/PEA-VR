#!/usr/bin/env python3
"""Freeze the class-disjoint LongEnough offset-0 learning protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import sys

_SCRIPTS = Path(__file__).resolve().parent.parent
for _d in (_SCRIPTS / "data", _SCRIPTS / "experiments"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from dataset_paths import LONGENOUGH_CACHE, DATA as DATA_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet-index",
        type=Path,
        default=LONGENOUGH_CACHE / "index.csv",
    )
    parser.add_argument(
        "--packet-metadata",
        type=Path,
        default=LONGENOUGH_CACHE / "metadata.json",
    )
    parser.add_argument(
        "--source-split",
        type=Path,
        default=DATA_ROOT / "longenough_smoke_v1/class_split.json",
    )
    parser.add_argument(
        "--mode-labels",
        type=Path,
        default=DATA_ROOT / "longenough_offset0_abr_modes_v1" / "trajectory_mode_assignments.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "longenough_offset0_protocol_v1",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalized_split(path: Path) -> dict[str, object]:
    source = json.loads(path.read_text(encoding="utf-8"))
    split = {
        "schema_version": 1,
        "split_seed": int(source["split_seed"]),
        "base_classes": sorted(int(value) for value in source["base_classes"]),
        "validation_classes": sorted(
            int(value) for value in source["validation_classes"]
        ),
        "test_classes": sorted(int(value) for value in source["test_classes"]),
    }
    role_sets = [
        set(split["base_classes"]),
        set(split["validation_classes"]),
        set(split["test_classes"]),
    ]
    if role_sets[0] & role_sets[1] or role_sets[0] & role_sets[2] or role_sets[1] & role_sets[2]:
        raise ValueError("Class roles overlap")
    if set.union(*role_sets) != set(range(100)):
        raise ValueError("Class split must cover video IDs 0..99 exactly once")
    split["counts"] = {
        "base": len(role_sets[0]),
        "validation": len(role_sets[1]),
        "test": len(role_sets[2]),
    }
    return split


def main() -> None:
    args = parse_args()
    split = normalized_split(args.source_split)
    role_by_video = {
        video: role
        for role, key in (
            ("base", "base_classes"),
            ("validation", "validation_classes"),
            ("test", "test_classes"),
        )
        for video in split[key]
    }

    modes = read_csv(args.mode_labels)
    mode_by_key: dict[tuple[str, int, int, int], tuple[int, str]] = {}
    for row in modes:
        key = (
            row["condition"],
            int(row["video"]),
            int(row["offset_min"]),
            int(row["repeat"]),
        )
        if key in mode_by_key:
            raise ValueError(f"Duplicate ABR-mode key: {key}")
        mode_by_key[key] = (int(row["abr_mode"]), row["abr_mode_name"])

    packet_rows = read_csv(args.packet_index)
    manifest: list[dict[str, object]] = []
    shard_position: dict[str, int] = {}
    keys: set[tuple[str, int, int, int]] = set()
    for sample_index, row in enumerate(packet_rows):
        condition = row["condition"]
        video = int(row["video"])
        offset = int(row["offset_min"])
        repeat = int(row["repeat"])
        key = (condition, video, offset, repeat)
        if key in keys:
            raise ValueError(f"Duplicate packet-cache key: {key}")
        keys.add(key)
        if key not in mode_by_key:
            raise ValueError(f"Missing ABR mode for {key}")
        mode, mode_name = mode_by_key[key]
        run_index = shard_position.get(condition, 0)
        shard_position[condition] = run_index + 1
        manifest.append(
            {
                "sample_index": sample_index,
                "split": role_by_video[video],
                "condition": condition,
                "cache_shard": f"{condition}.npz",
                "cache_run_index": run_index,
                "video": video,
                "offset_min": offset,
                "repeat": repeat,
                "run_id": row["run_id"],
                "abr_mode": mode,
                "abr_mode_name": mode_name,
                "packet_count": int(row["packet_count"]),
                "total_bytes": int(row["total_bytes"]),
                "window_packet_sha256": row["window_packet_sha256"],
            }
        )

    if len(manifest) != 4000 or len(keys) != 4000:
        raise ValueError(f"Expected 4,000 unique runs, found {len(manifest)}")
    if set(mode_by_key) != keys:
        raise ValueError("Packet cache and ABR-mode tables do not have identical keys")
    if shard_position != {condition: 1000 for condition in ("bw1", "bw2", "bw4", "bw8")}:
        raise ValueError(f"Unexpected condition counts: {shard_position}")
    for video in range(100):
        selected = [row for row in manifest if int(row["video"]) == video]
        if len(selected) != 40 or {row["split"] for row in selected} != {role_by_video[video]}:
            raise ValueError(f"Video {video} does not have 40 runs in one split")
        if {int(row["abr_mode"]) for row in selected} != {0, 1, 2}:
            raise ValueError(f"Video {video} does not cover all three ABR modes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.output_dir / "class_split.json"
    split_path.write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = args.output_dir / "run_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    metadata = {
        "schema": "longenough-offset0-class-disjoint-protocol",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "few-shot video identity recognition under ABR multimodality",
        "input": "encrypted packet timestamp, direction, and length only",
        "selection": {"offset_min": 0, "window_sec": 60, "conditions": ["bw1", "bw2", "bw4", "bw8"]},
        "class_split": split,
        "runs": len(manifest),
        "runs_by_split": {
            role: sum(row["split"] == role for row in manifest)
            for role in ("base", "validation", "test")
        },
        "mode_label_policy": {
            "base": "allowed only as auxiliary supervision or sampling metadata",
            "validation": "evaluation stratification only",
            "test": "evaluation stratification only",
            "model_input": False,
            "interpretation": "coarse QoE-derived trajectory archetype, not a ground-truth controller state",
        },
        "external_fixed100_policy": "excluded; reserved for a later external-domain evaluation",
        "provenance": {
            "packet_index": str(args.packet_index),
            "packet_index_sha256": sha256_file(args.packet_index),
            "packet_metadata": str(args.packet_metadata),
            "packet_metadata_sha256": sha256_file(args.packet_metadata),
            "source_split": str(args.source_split),
            "source_split_sha256": sha256_file(args.source_split),
            "mode_labels": str(args.mode_labels),
            "mode_labels_sha256": sha256_file(args.mode_labels),
            "frozen_split_sha256": sha256_file(split_path),
            "run_manifest_sha256": sha256_file(manifest_path),
        },
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

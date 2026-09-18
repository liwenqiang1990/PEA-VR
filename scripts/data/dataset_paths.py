#!/usr/bin/env python3
"""Dataset and output locations, resolved in one place for every script.

Neither dataset ships with this repository; both are large and one is a third-party release.
By default the scripts look for them under `datasets/` at the repository root, which is ignored
by git, so you can either place (or symlink) the data there:

    datasets/LongEnough_offset0_60s_packet_cache_v1/
    datasets/mobile_yt_dataset/

or point these environment variables anywhere else:

    PEA_VR_LONGENOUGH_CACHE   the LongEnough offset-0 60 s packet cache directory
    PEA_VR_YDMS_ROOT          the YDMS release root (its `mobile_yt_dataset` directory)
    PEA_VR_DATA               where scripts write run outputs (default: `runs/`)

Every path is anchored at the repository root, so the scripts behave the same whatever the working
directory is. Scripts that also take a path on the command line let that argument win.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = REPO_ROOT / "datasets"

LONGENOUGH_CACHE = Path(os.environ.get(
    "PEA_VR_LONGENOUGH_CACHE", DATASETS / "LongEnough_offset0_60s_packet_cache_v1"))

YDMS_ROOT = Path(os.environ.get("PEA_VR_YDMS_ROOT", DATASETS / "mobile_yt_dataset"))

# Output root. Named `runs` rather than `data` so it is not confused with `scripts/data/`,
# which holds the data-preparation code.
DATA = Path(os.environ.get("PEA_VR_DATA", REPO_ROOT / "runs"))


def require(path: Path, what: str) -> Path:
    """Fail with an actionable message instead of deep inside a loader."""
    if not path.exists():
        raise SystemExit(
            f"{what} not found at: {path}\n"
            f"Place the data under {DATASETS}/ or set the matching environment variable "
            f"(see scripts/data/dataset_paths.py).")
    return path

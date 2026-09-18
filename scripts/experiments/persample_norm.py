#!/usr/bin/env python3
"""Per-sample input standardization, as a candidate fix for the cross-bandwidth collapse.

Finding that motivates this:
on YDMS, PEA-VR scores 32.80 / 37.76 on the two cross-bandwidth scenarios while the reproduced
Gansekoele baseline scores 64.78 / 60.69. The two differ in how the input is scaled --
`normalize_from_base` applies log1p plus base-split z-scores, keeping each session's ABSOLUTE
byte scale, whereas Gansekoele "standardized each sample individually" (its Sec. IV), which
removes the bandwidth-induced scale factor that cross-bandwidth matching has to see past.

Three modes:
  base       -- unchanged `normalize_from_base` (the current PEA-VR input)
  persample  -- each (sample, scale, feature) z-scored over its own time axis
  both       -- the two representations concatenated along the feature axis (in_dim doubles),
                so absolute scale stays available where it helps and a scale-free view is added

`both` is the variant that can win on cross-bandwidth without giving up whatever the absolute
scale contributes elsewhere. Callers import from here.
"""
from __future__ import annotations
import numpy as np


def per_sample_standardize(arrays: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """z-score every (sample, feature) series over the time axis of each scale."""
    out = {}
    for scale, a in arrays.items():
        v = a.astype(np.float64)
        mu = v.mean(axis=1, keepdims=True)
        sd = v.std(axis=1, keepdims=True)
        out[scale] = ((v - mu) / np.maximum(sd, 1e-8)).astype(np.float32)
    return out


def apply_normalization(data, mode: str, normalize_from_base):
    """Return `data` with `arrays` set according to `mode`; also returns the input dimension."""
    if mode not in {"base", "persample", "both"}:
        raise ValueError(f"unknown normalization mode: {mode}")
    raw = {s: np.asarray(a).copy() for s, a in data.arrays.items()}
    in_dim = next(iter(raw.values())).shape[2]
    if mode == "base":
        return normalize_from_base(data), in_dim
    ps = per_sample_standardize(raw)
    if mode == "persample":
        data.arrays = ps
        data.normalization = {"transform": "per-sample z-score over the time axis"}
        return data, in_dim
    data.arrays = raw                      # normalize_from_base mutates in place
    base = normalize_from_base(data)
    merged = {s: np.concatenate((base.arrays[s], ps[s]), axis=2) for s in raw}
    base.arrays = merged
    base.normalization = {"transform": "base-split log1p z-score CONCAT per-sample z-score",
                          "in_dim": 2 * in_dim}
    return base, 2 * in_dim

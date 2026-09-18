#!/usr/bin/env python3
"""YDMS reconstruction, step 2: D=60 wall-clock multiscale views (native payload, TCP+UDP combined).

Replicates the LongEnough AMP input exactly so the same encoder consumes YDMS:
  scales 100/500/2000 ms -> 600/120/30 bins; 5 features per bin
  [down_bytes, up_bytes, down_packets, up_packets, down_mean_packet_length]
  down = direction 0 (remote->local, content), up = direction 1. Combined TCP+UDP. Native payload.
Only runs with full duration >= 60 s are kept (short runs excluded, never padded).

Also emits the observation-window bandwidth CONDITION sidecar (never model input):
  obs_first_bw, obs_min_bw (bottleneck; the chosen level condition), obs_max_bw, is_dynamic,
  obs_min_group in {low<=1000, mid1001-2500, high2501-399999, unlimited_400000}.

Reads the Part-A packet NPZ (mmap) + re-reads each run's first timestamp and bw_settings.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import numpy as np

import sys

_SCRIPTS = Path(__file__).resolve().parent.parent
for _d in (_SCRIPTS / "data", _SCRIPTS / "experiments"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from dataset_paths import YDMS_ROOT, DATA as DATA_ROOT

DATA = DATA_ROOT / "ydms_external_v1"
ROOT = YDMS_ROOT / "dataset"
SCALES = (100, 500, 2000)
FINEST = 100
FINE_BINS = 600            # 60000 / 100
D_NS = 60_000_000_000
FEATURES = ["down_bytes", "up_bytes", "down_packets", "up_packets", "down_mean_packet_length"]


def obs_group(b: float) -> str:
    if b == 400000: return "unlimited_400000"
    if b <= 1000: return "low_le_1000"
    if b <= 2500: return "mid_1001_2500"
    if b < 400000: return "high_2501_399999"
    return "other"


def first_ts(run_id: str) -> float | None:
    p = ROOT / run_id / "video_traffic.csv"
    try:
        with p.open() as f:
            f.readline()
            return float(f.readline().split(",")[0])
    except (OSError, ValueError, IndexError):
        return None


def read_bw(run_id: str):
    rows = []
    try:
        with (ROOT / run_id / "bw_settings.csv").open() as f:
            for r in csv.DictReader(f):
                try: rows.append((float(r["timestamp"]), float(r["bandwidth"])))
                except (KeyError, TypeError, ValueError): pass
    except (OSError, csv.Error):
        pass
    return sorted(rows)


def obs_bandwidth(run_id: str):
    """first/min/max bw active during [t0, t0+60s), plus is_dynamic."""
    t0 = first_ts(run_id); bw = read_bw(run_id)
    if t0 is None or not bw:
        return None
    active = bw[0][1]
    for ts, b in bw:
        if ts <= t0: active = b
        else: break
    window = [active]
    dyn = False
    for ts, b in bw:
        if t0 < ts <= t0 + 60:
            window.append(b); dyn = True
    return dict(obs_first_bw=active, obs_min_bw=float(min(window)),
                obs_max_bw=float(max(window)), is_dynamic=dyn)


def aggregate(fine: np.ndarray, factor: int) -> np.ndarray:
    g = fine.reshape(fine.shape[0] // factor, factor, fine.shape[1])
    out = np.zeros((g.shape[0], fine.shape[1]), dtype=np.float32)
    out[:, :4] = g[:, :, :4].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dm = np.where(out[:, 2] > 0, out[:, 0] / out[:, 2], 0.0)
    out[:, 4] = dm
    return out


def build_fine(times, dirs, plens) -> np.ndarray:
    m = times < D_NS
    times, dirs, plens = times[m], dirs[m], plens[m].astype(np.float64)
    bins = np.clip((times // (FINEST * 1_000_000)).astype(np.int64), 0, FINE_BINS - 1)
    down = dirs == 0; up = dirs == 1
    fine = np.zeros((FINE_BINS, 5), dtype=np.float32)
    fine[:, 0] = np.bincount(bins[down], weights=plens[down], minlength=FINE_BINS)
    fine[:, 1] = np.bincount(bins[up], weights=plens[up], minlength=FINE_BINS)
    fine[:, 2] = np.bincount(bins[down], minlength=FINE_BINS)
    fine[:, 3] = np.bincount(bins[up], minlength=FINE_BINS)
    return fine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", type=Path, default=DATA / "ydms_packets_120s_v1.npz")
    ap.add_argument("--conditions", type=Path, default=DATA / "ydms_conditions_120s_v1.csv")
    ap.add_argument("--output", type=Path, default=DATA / "ydms_views_60s_v1.npz")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    z = np.load(args.packets, allow_pickle=False, mmap_mode="r")
    ro = np.asarray(z["run_offsets"]); rid = z["run_id"]; vid = z["video_id"]; lab = z["label"]
    T = z["packet_time_relative_ns"]; Dd = z["packet_direction"]; P = z["packet_payload_length"]
    dur = {r["canonical_sample_id"]: float(r["duration_s"])
           for r in csv.DictReader(open(args.conditions))}

    keep = [i for i in range(len(ro) - 1) if dur.get(str(i), 0.0) >= 60.0]
    if args.limit:
        keep = keep[: args.limit]
    print(f"[views] {len(keep)} samples with duration>=60s")

    views = {s: np.zeros((len(keep), 60_000 // s, 5), dtype=np.float32) for s in SCALES}
    labels = np.zeros(len(keep), dtype=np.int32)
    vids = []; rids = []; csid = np.zeros(len(keep), dtype=np.int32)
    cond_rows = []
    dropped_bw = 0
    for k, i in enumerate(keep):
        a, b = int(ro[i]), int(ro[i + 1])
        fine = build_fine(np.asarray(T[a:b]), np.asarray(Dd[a:b]), np.asarray(P[a:b]))
        for s in SCALES:
            views[s][k] = aggregate(fine, s // FINEST)
        labels[k] = int(lab[i]); csid[k] = i
        rid_k = str(rid[i]); vids.append(str(vid[i])); rids.append(rid_k)
        ob = obs_bandwidth(rid_k)
        if ob is None:
            dropped_bw += 1
            ob = dict(obs_first_bw=float("nan"), obs_min_bw=float("nan"),
                      obs_max_bw=float("nan"), is_dynamic=False)
        cond_rows.append((i, rid_k, str(vid[i]), int(lab[i]),
                          ob["obs_first_bw"], ob["obs_min_bw"], ob["obs_max_bw"],
                          int(ob["is_dynamic"]), obs_group(ob["obs_min_bw"])))
        if (k + 1) % 1000 == 0:
            print(f"  {k+1}/{len(keep)}")

    out = {f"view_{s}": views[s] for s in SCALES}
    out.update(label=labels, video_id=np.array(vids), run_id=np.array(rids),
               canonical_sample_id=csid, scales=np.array(SCALES),
               feature_names=np.array(FEATURES))
    outp = args.output if not args.limit else args.output.with_name(args.output.stem + "_smoke.npz")
    outp.parent.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(outp, **out)

    condp = (DATA / "ydms_conditions_60s_v1.csv") if not args.limit else (DATA / "ydms_conditions_60s_smoke.csv")
    with condp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["canonical_sample_id", "run_id", "video_id", "label",
                    "obs_first_bw", "obs_min_bw", "obs_max_bw", "is_dynamic", "obs_min_group"])
        w.writerows(cond_rows)

    from collections import Counter
    manifest = {
        "views_npz": str(outp), "conditions": str(condp),
        "samples": len(keep), "identities": len(set(vids)),
        "dropped_bw_meta": dropped_bw,
        "obs_min_group_counts": dict(Counter(r[8] for r in cond_rows)),
        "is_dynamic_frac": round(float(np.mean([r[7] for r in cond_rows])), 3) if cond_rows else 0.0,
        "view_shapes": {f"view_{s}": list(views[s].shape) for s in SCALES},
    }
    (DATA / ("views_manifest_60s_smoke.json" if args.limit else "views_manifest_60s_v1.json")).write_text(
        json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""YDMS reconstruction, step 1: lossless packet-level NPZ from raw video_traffic.csv.

Per run (= one canonical video session = one sample) we build a faithful ordered packet stream:
  packet_time_relative_ns  int64   ns from the FIRST video_traffic packet (startup kept)
  packet_direction         uint8   0 = downstream (remote->local, content), 1 = upstream
  packet_payload_length    uint16  tcpLen + udpLen  (RAW native payload; NO harmonization here)
  packet_protocol          uint8   6 = TCP, 17 = UDP
Runs are concatenated; run i occupies run_offsets[i]:run_offsets[i+1]. Sidecars (video_id, bandwidth
stratum, QoE) are kept SEPARATE so they can never leak into model input.

Dedup BEFORE anything downstream: hash the canonical packet sequence; keep one per within-label
group; drop all-zero runs; drop cross-label conflicting hash groups.

Length harmonization and TCP/UDP splitting are deferred to the feature layer (protocol decision
2026-08-16): this NPZ stores RAW native payload lengths and one tagged stream.
"""
from __future__ import annotations

import argparse, csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

import sys

_SCRIPTS = Path(__file__).resolve().parent.parent
for _d in (_SCRIPTS / "data", _SCRIPTS / "experiments"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from dataset_paths import YDMS_ROOT, DATA as DATA_ROOT

DEFAULT_ROOT = YDMS_ROOT
DEFAULT_OUT = DATA_ROOT / "ydms_external_v1"


def is_private(ip: str) -> bool:
    try:
        a, b, *_ = (int(x) for x in ip.split("."))
    except (ValueError, IndexError):
        return False
    return a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168)


def bandwidth_group(value: float) -> str:
    if value == 400000:
        return "unlimited_400000"
    if value <= 1000:
        return "low_le_1000"
    if value <= 2500:
        return "mid_1001_2500"
    if value < 400000:
        return "high_2501_399999"
    return "other"


def first_nonempty_videoid(app_path: Path) -> str:
    try:
        with app_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                vid = (row.get("videoid") or "").strip()
                if vid:
                    return vid
    except (OSError, UnicodeError, csv.Error):
        pass
    return ""


def read_bandwidth(path: Path) -> list[tuple[float, float]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    rows.append((float(row["timestamp"]), float(row["bandwidth"])))
                except (KeyError, TypeError, ValueError):
                    continue
    except (OSError, UnicodeError, csv.Error):
        return []
    return sorted(rows)


def bw_stratum(settings, start, end) -> tuple[str, float]:
    """Representative stratum from bandwidths active during [start,end]."""
    if not settings:
        return "unknown", float("nan")
    active_i = 0
    for i, (ts, _) in enumerate(settings):
        if ts <= start:
            active_i = i
        else:
            break
    vals = [settings[active_i][1]]
    for ts, bw in settings[active_i + 1:]:
        if ts <= end:
            vals.append(bw)
        else:
            break
    med = float(np.median(vals))
    return bandwidth_group(med), med


def parse_run(run_dir: Path, cap_ns: int | None = None):
    """Return dict(packets arrays, video_id, bw) or None if unusable.

    cap_ns: if set, keep only packets within [0, cap_ns) from the first packet (space saver;
    raw CSV on disk remains the true lossless archive, so a larger cap is always rebuildable).
    """
    vt = run_dir / "video_traffic.csv"
    if not vt.is_file():
        return None
    times, dirs, plens, protos = [], [], [], []
    ip_count: Counter = Counter()
    rows_raw = []
    try:
        with vt.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    t = float(row["timestamp"])
                except (KeyError, TypeError, ValueError):
                    continue
                src = (row.get("ipSrc") or "").strip()
                dst = (row.get("ipDst") or "").strip()
                tl = int(row.get("tcpLen") or 0 or 0) if (row.get("tcpLen") or "").strip() else 0
                ul = int(row.get("udpLen") or 0) if (row.get("udpLen") or "").strip() else 0
                proto = int(row.get("payloadProtocolNumber") or 0) if (row.get("payloadProtocolNumber") or "").strip() else 0
                rows_raw.append((t, src, dst, tl + ul, proto))
                if is_private(src):
                    ip_count[src] += 1
                if is_private(dst):
                    ip_count[dst] += 1
    except (OSError, UnicodeError, csv.Error):
        return None
    if not rows_raw:
        return None
    client_ip = ip_count.most_common(1)[0][0] if ip_count else None

    rows_raw.sort(key=lambda r: r[0])            # order by time
    t0 = rows_raw[0][0]
    for t, src, dst, payload, proto in rows_raw:
        # direction: 0 = downstream (remote->local, content), 1 = upstream (local->remote)
        if client_ip is not None and src == client_ip:
            d = 1
        elif client_ip is not None and dst == client_ip:
            d = 0
        else:
            d = 0                                 # fallback: treat as downstream
        times.append(int(round((t - t0) * 1e9)))
        dirs.append(d)
        plens.append(min(payload, 65535))
        protos.append(proto)

    times = np.asarray(times, dtype=np.int64)
    dirs = np.asarray(dirs, dtype=np.uint8)
    plens = np.asarray(plens, dtype=np.uint16)
    protos = np.asarray(protos, dtype=np.uint8)

    if cap_ns is not None:
        m = times < cap_ns
        times, dirs, plens, protos = times[m], dirs[m], plens[m], protos[m]
    if len(times) == 0:
        return None

    settings = read_bandwidth(run_dir / "bw_settings.csv")
    strat, bw_med = bw_stratum(settings, t0, rows_raw[-1][0])
    vid = first_nonempty_videoid(run_dir / "application_data.csv")

    seq_hash = hashlib.sha256(
        times.tobytes() + dirs.tobytes() + plens.tobytes() + protos.tobytes()
    ).hexdigest()
    all_zero = bool(plens.sum() == 0)
    return dict(times=times, dirs=dirs, plens=plens, protos=protos,
                video_id=vid, bw_stratum=strat, bw_median=bw_med,
                seq_hash=seq_hash, all_zero=all_zero, n=len(times),
                duration_s=float((rows_raw[-1][0] - t0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="smoke: only first N runs")
    ap.add_argument("--cap-seconds", type=float, default=None,
                    help="keep only first N seconds per run (space saver; raw CSV stays the archive)")
    args = ap.parse_args()
    cap_ns = int(args.cap_seconds * 1e9) if args.cap_seconds else None
    tag = (f"{int(args.cap_seconds)}s_" if args.cap_seconds else "") + ("smoke" if args.limit else "v1")

    src_root = args.dataset_root / "dataset"
    run_dirs = sorted([p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("run_")],
                      key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0)
    if args.limit:
        run_dirs = run_dirs[: args.limit]
    print(f"[ydms] {len(run_dirs)} run dirs to parse")

    parsed = []
    empty_vid = 0
    for i, rd in enumerate(run_dirs):
        r = parse_run(rd, cap_ns)
        if r is None:
            continue
        r["run_id"] = rd.name
        if not r["video_id"]:
            empty_vid += 1
        parsed.append(r)
        if (i + 1) % 500 == 0:
            print(f"  parsed {i+1}/{len(run_dirs)}")
    print(f"[ydms] parsed {len(parsed)} runs; empty_video_id={empty_vid}")

    # ---- dedup: within-label exact seq duplicates; drop all-zero; drop cross-label hash groups ----
    by_hash: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(parsed):
        by_hash[r["seq_hash"]].append(idx)
    keep = np.ones(len(parsed), dtype=bool)
    dup_removed = zero_removed = conflict_removed = 0
    for h, idxs in by_hash.items():
        vids = {parsed[j]["video_id"] for j in idxs}
        if len(vids) > 1:                          # cross-label conflict -> drop all
            for j in idxs:
                keep[j] = False
            conflict_removed += len(idxs)
            continue
        for j in idxs[1:]:                         # keep first canonical
            keep[j] = False; dup_removed += 1
    for idx, r in enumerate(parsed):
        if keep[idx] and (r["all_zero"] or not r["video_id"]):
            keep[idx] = False; zero_removed += 1
    canon = [parsed[i] for i in range(len(parsed)) if keep[i]]
    print(f"[ydms] canonical={len(canon)}  dup_removed={dup_removed} "
          f"zero/empty_removed={zero_removed} conflict_removed={conflict_removed}")

    # ---- assemble concatenated arrays ----
    vids_sorted = sorted({r["video_id"] for r in canon})
    vid2label = {v: i for i, v in enumerate(vids_sorted)}
    offs = np.zeros(len(canon) + 1, dtype=np.int64)
    for i, r in enumerate(canon):
        offs[i + 1] = offs[i] + r["n"]
    T = np.concatenate([r["times"] for r in canon]) if canon else np.zeros(0, np.int64)
    D = np.concatenate([r["dirs"] for r in canon]) if canon else np.zeros(0, np.uint8)
    P = np.concatenate([r["plens"] for r in canon]) if canon else np.zeros(0, np.uint16)
    R = np.concatenate([r["protos"] for r in canon]) if canon else np.zeros(0, np.uint8)

    args.output.mkdir(parents=True, exist_ok=True)
    npz_path = args.output / f"ydms_packets_{tag}.npz"
    np.savez(npz_path, schema_version=np.uint16(1),
             packet_time_relative_ns=T, packet_direction=D,
             packet_payload_length=P, packet_protocol=R, run_offsets=offs,
             run_id=np.array([r["run_id"] for r in canon]),
             video_id=np.array([r["video_id"] for r in canon]),
             label=np.array([vid2label[r["video_id"]] for r in canon], dtype=np.int32),
             canonical_sample_id=np.arange(len(canon), dtype=np.int32),
             seq_sha256=np.array([r["seq_hash"] for r in canon]))
    # condition sidecar (never model input)
    sidecar = args.output / f"ydms_conditions_{tag}.csv"
    with sidecar.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["canonical_sample_id", "run_id", "video_id", "label",
                    "bw_stratum", "bw_median", "n_packets", "duration_s"])
        for i, r in enumerate(canon):
            w.writerow([i, r["run_id"], r["video_id"], vid2label[r["video_id"]],
                        r["bw_stratum"], r["bw_median"], r["n"], round(r["duration_s"], 3)])

    # manifest
    label_counts = Counter(r["video_id"] for r in canon)
    ge11 = sum(1 for v, c in label_counts.items() if c >= 11)
    strat_counts = Counter(r["bw_stratum"] for r in canon)
    manifest = {
        "npz": str(npz_path), "sidecar": str(sidecar),
        "cap_seconds": args.cap_seconds,
        "runs_parsed": len(parsed), "canonical_samples": len(canon),
        "identities": len(vids_sorted), "identities_ge11": ge11,
        "dedup": {"dup_removed": dup_removed, "zero_or_empty_removed": zero_removed,
                  "conflict_removed": conflict_removed},
        "bw_stratum_counts": dict(strat_counts),
        "total_packets": int(len(T)),
        "duration_s": {"median": float(np.median([r["duration_s"] for r in canon])) if canon else 0,
                       "p90": float(np.percentile([r["duration_s"] for r in canon], 90)) if canon else 0,
                       "reach_60s": int(sum(1 for r in canon if r["duration_s"] >= 60))},
    }
    (args.output / ("build_manifest_smoke.json" if args.limit else "build_manifest_v1.json")).write_text(
        json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

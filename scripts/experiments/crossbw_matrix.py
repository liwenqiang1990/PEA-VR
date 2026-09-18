#!/usr/bin/env python3
"""Reference x query bandwidth retrieval matrix for the cross-asymmetry figure.

Inference only, on the three final d256/e512 PEA-VR checkpoints. For every ordered pair of
LongEnough bandwidth conditions (bw1, bw2, bw4, bw8) an episode set is built with the support
sessions drawn from the REFERENCE condition and the query sessions from the QUERY condition; the
diagonal is the same-condition case. A pooled-reference row draws support from all four conditions.

All 30 test identities are usable because every identity has 10 sessions per condition, so the
matrix is far better powered than the YDMS cross-bandwidth column (8 usable identities).

Writes `data/baselines_lit_v1/crossbw_matrix.json` with per-seed values.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_SCRIPTS = Path(__file__).resolve().parent.parent
for _d in (_SCRIPTS / "data", _SCRIPTS / "experiments"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from dataset_paths import LONGENOUGH_CACHE, DATA as DATA_ROOT

from longenough_offset0_multiscale import (FEATURE_NAMES, load_dynamic_multiscale,
                                           normalize_from_base, parse_scales)
from train_longenough_offset0_multiscale_baseline import choose_device
from train_longenough_offset0_swapcanonical import AMPEncoder, extract_embeddings

CACHE = LONGENOUGH_CACHE
PROTO = DATA_ROOT / "longenough_offset0_protocol_v1"
E2 = DATA_ROOT / "abr_essence_explore_v1"
LIT = DATA_ROOT / "baselines_lit_v1"
SCALES = (100, 500, 2000)
SEEDS = (20260814, 20260815, 20260816)
COND = {0: "bw1", 1: "bw2", 2: "bw4", 3: "bw8"}


def build(video, cond, split, ref, qry, n_way, k, q, n, seed):
    """`ref=None` pools the support over all conditions."""
    pool = np.flatnonzero(split == "test")
    by = {}
    for i in pool:
        by.setdefault(int(video[i]), []).append(int(i))
    rng = np.random.RandomState(seed)
    usable = []
    for v, s in by.items():
        a = np.array(s)
        ok_r = len(a) >= k if ref is None else (cond[a] == ref).sum() >= k
        ok_q = (cond[a] == qry).sum() >= q
        if ok_r and ok_q:
            usable.append(v)
    eps = []
    for _ in range(n):
        if len(usable) < n_way:
            break
        classes = rng.choice(usable, size=n_way, replace=False)
        sup, que = [], []
        for c in classes:
            a = np.array(by[int(c)])
            qp = a[cond[a] == qry]
            chosen_q = rng.choice(qp, size=q, replace=False)
            rp = a if ref is None else a[cond[a] == ref]
            rp = np.setdiff1d(rp, chosen_q)          # support and query stay disjoint
            if len(rp) < k:
                break
            sup += rng.choice(rp, size=k, replace=False).tolist()
            que += chosen_q.tolist()
        if len(sup) == n_way * k:
            eps.append({"support_indices": sup, "query_indices": que})
    return eps, len(usable)


def supportmax(emb, video, eps):
    correct = []
    for ep in eps:
        s = np.asarray(ep["support_indices"]); q = np.asarray(ep["query_indices"])
        classes = np.unique(video[s])
        cols = [np.max(emb[q] @ emb[s[video[s] == c]].T, axis=1) for c in classes]
        correct.append(classes[np.argmax(np.column_stack(cols), axis=1)] == video[q])
    return float(np.concatenate(correct).mean() * 100) if correct else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--shot", type=int, default=2)
    p.add_argument("--query", type=int, default=2)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = choose_device(args.device)
    data = normalize_from_base(load_dynamic_multiscale(CACHE, PROTO / "run_manifest.csv",
                                                       parse_scales("100,500,2000")))
    video = np.asarray(data.video); cond = np.asarray(data.condition); split = np.asarray(data.split)

    pairs = [(r, q) for r in COND for q in COND] + [(None, q) for q in COND]
    eps = {}
    for r, q in pairs:
        e, n_use = build(video, cond, split, r, q, args.n_way, args.shot, args.query,
                         args.episodes, seed=20260903 + (99 if r is None else r) * 10 + q)
        key = f"{'pooled' if r is None else COND[r]}->{COND[q]}"
        eps[key] = e
        print(f"  {key:<16} {len(e):4d} episodes over {n_use} identities", flush=True)

    out: dict[str, list[float]] = {}
    for seed in SEEDS:
        st = torch.load(E2 / f"e2_cap_d256e512_s{str(seed)[-3:]}" / "best.pt", map_location=device)
        m = AMPEncoder(scales=SCALES, in_dim=len(FEATURE_NAMES), tokens=30, d_model=256,
                                 style_dim=16, embedding_dim=512, transformer_layers=2,
                                 num_base_classes=len(st["base_classes"]), dropout=0.1,
                                 content_input="observed", temporal_encoder="identity").to(device)
        m.load_state_dict(st["model_state"]); m.eval()
        emb = extract_embeddings(m, data, np.arange(len(video)), SCALES, device, 128, 0)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        for key, e in eps.items():
            out.setdefault(key, []).append(round(supportmax(emb, video, e), 2))
        print(f"[seed {seed}] done", flush=True)

    summary = {k: {"per_seed": v, "mean": round(float(np.mean(v)), 2),
                   "std": round(float(np.std(v, ddof=1)), 2)} for k, v in out.items()}
    LIT.mkdir(parents=True, exist_ok=True)
    (LIT / "crossbw_matrix.json").write_text(json.dumps(summary, indent=2) + "\n")
    names = list(COND.values())
    print(f"\n{'ref \\ query':<12}" + "".join(f"{n:>14}" for n in names))
    for r in names + ["pooled"]:
        print(f"{r:<12}" + "".join(
            f"{summary[f'{r}->{c}']['mean']:>8.2f}±{summary[f'{r}->{c}']['std']:<5.2f}"
            for c in names))
    print(f"\nwritten -> {LIT / 'crossbw_matrix.json'}")


if __name__ == "__main__":
    main()

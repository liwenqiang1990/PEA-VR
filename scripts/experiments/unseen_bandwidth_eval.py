#!/usr/bin/env python3
"""Unseen-bandwidth evaluation: episodes drawn entirely from a condition held out of training.

Distinct from the cross-bandwidth column of the overall table. There, all four LongEnough
conditions appear in encoder training and the test measures matching ACROSS conditions. Here the
held-out condition contributed neither training samples nor normalization statistics
(`--exclude-condition` in `e2_train_repr.py`), so the question is generalization to a bandwidth
the encoder has never seen.

Episodes: 5-way, 2 support and 2 query sessions, ALL drawn from the held-out condition, over the
30 test identities -- every identity has 10 sessions per condition, so all 30 are usable.
Scored with SupportMax from the held-out checkpoint of each seed.

Writes `data/baselines_lit_v1/unseen_bandwidth.json`.
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
COND_NAME = {0: "bw1", 1: "bw2", 2: "bw4", 3: "bw8"}


def episodes_within(video, cond, split, held, n_way, k, q, n, seed):
    pool = np.flatnonzero((split == "test") & (cond == held))
    by = {}
    for i in pool:
        by.setdefault(int(video[i]), []).append(int(i))
    usable = [v for v, s in by.items() if len(s) >= k + q]
    rng = np.random.RandomState(seed)
    eps = []
    for _ in range(n):
        if len(usable) < n_way:
            break
        classes = rng.choice(usable, size=n_way, replace=False)
        sup, qry = [], []
        for c in classes:
            sel = rng.permutation(by[int(c)])
            sup += sel[:k].tolist(); qry += sel[k:k + q].tolist()
        eps.append({"episode": len(eps), "support_indices": sup, "query_indices": qry})
    print(f"  held-out {COND_NAME[held]}: {len(eps)} episodes over {len(usable)} usable identities",
          flush=True)
    return eps


def supportmax(emb, video, eps):
    z = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    correct = []
    for ep in eps:
        s = np.asarray(ep["support_indices"]); q = np.asarray(ep["query_indices"])
        classes = np.unique(video[s])
        cols = [np.max(z[q] @ z[s[video[s] == c]].T, axis=1) for c in classes]
        correct.append(classes[np.argmax(np.column_stack(cols), axis=1)] == video[q])
    return float(np.concatenate(correct).mean() * 100)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--held", type=int, nargs="+", default=[0, 3])
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--shot", type=int, default=2)
    p.add_argument("--query", type=int, default=2)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = choose_device(args.device)
    raw = load_dynamic_multiscale(CACHE, PROTO / "run_manifest.csv", parse_scales("100,500,2000"))
    video = np.asarray(raw.video); cond = np.asarray(raw.condition); split = np.asarray(raw.split)
    out: dict[str, dict[str, list[float]]] = {}

    for held in args.held:
        eps = episodes_within(video, cond, split, held, args.n_way, args.shot, args.query,
                              args.episodes, seed=20260902 + held)
        if not eps:
            continue
        name = COND_NAME[held]

        # PEA-VR: its normalization is fit on base minus the held-out condition, so rebuild it
        data = load_dynamic_multiscale(CACHE, PROTO / "run_manifest.csv",
                                       parse_scales("100,500,2000"))
        drop = (np.asarray(data.split) == "base") & (np.asarray(data.condition) == held)
        data.split = np.where(drop, "excluded", data.split)
        data = normalize_from_base(data)
        for seed in SEEDS:
            ck = E2 / f"e2_ho{held}_d256e512_s{str(seed)[-3:]}" / "best.pt"
            if not ck.exists():
                print(f"!! missing {ck}"); continue
            st = torch.load(ck, map_location=device)
            m = AMPEncoder(scales=SCALES, in_dim=len(FEATURE_NAMES), tokens=30,
                                     d_model=256, style_dim=16, embedding_dim=512,
                                     transformer_layers=2,
                                     num_base_classes=len(st["base_classes"]), dropout=0.1,
                                     content_input="observed", temporal_encoder="identity").to(device)
            m.load_state_dict(st["model_state"]); m.eval()
            emb = extract_embeddings(m, data, np.arange(len(video)), SCALES, device, 128, 0)
            out.setdefault("PEA-VR", {}).setdefault(name, []).append(
                round(supportmax(emb, video, eps), 2))

    summary = {m: {k: {"per_seed": v, "mean": round(float(np.mean(v)), 2),
                       "std": round(float(np.std(v, ddof=1)), 2) if len(v) > 1 else 0.0}
                   for k, v in d.items()} for m, d in out.items()}
    LIT.mkdir(parents=True, exist_ok=True)
    (LIT / "unseen_bandwidth.json").write_text(json.dumps(summary, indent=2) + "\n")
    names = sorted({k for d in summary.values() for k in d})
    print(f"\n{'method':<12}" + "".join(f"{'held-out ' + n:>20}" for n in names))
    for m in summary:
        print(f"{m:<12}" + "".join(
            f"{summary[m][n]['mean']:>13.2f}±{summary[m][n]['std']:<6.2f}" if n in summary[m]
            else f"{'--':>20}" for n in names))
    print(f"\nwritten -> {LIT / 'unseen_bandwidth.json'}")


if __name__ == "__main__":
    main()

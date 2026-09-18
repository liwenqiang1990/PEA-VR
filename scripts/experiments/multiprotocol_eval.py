#!/usr/bin/env python3
"""Multi-protocol evaluation of the SAME frozen AMP embedding on LongEnough test identities.

Shows the embedding is a general video fingerprint, not only a few-shot trick. Protocols on the 30
UNSEEN test identities (no retraining; frozen paper checkpoints):
  1. few-shot        : novel-class N-way K-shot SupportMax (cross-mode + mixed), from locked episodes
  2. closed-set      : 30-way identification, gallery/query split per video (all-vs-all NN), + cross-mode
  3. open-set        : half the test videos enrolled (known), half unknown; AUROC(known/unknown) + OSCR
  4. clustering      : unsupervised KMeans(K=30) on embeddings, ARI / NMI vs true video
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent
for _d in (_SCRIPTS / "data", _SCRIPTS / "experiments"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from dataset_paths import LONGENOUGH_CACHE, DATA as DATA_ROOT

from longenough_offset0_multiscale import load_dynamic_multiscale, normalize_from_base, parse_scales
from train_longenough_offset0_multiscale_baseline import choose_device
from train_longenough_offset0_swapcanonical import (extract_embeddings as extract_standard,
                                                    load_episodes, load_swap)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score

CACHE = LONGENOUGH_CACHE
PROTO = DATA_ROOT / "longenough_offset0_protocol_v1"
CKPTS = {"20260814": DATA_ROOT / "longenough_offset0_full100_alignedpool_seed1_v1" / "best.pt",
         "20260815": DATA_ROOT / "longenough_offset0_full100_alignedpool_seed2_v1" / "best.pt"}
OUT = DATA_ROOT / "multiprotocol_v1/longenough_multiprotocol.json"


def fewshot(emb, video, episodes):
    corr = tot = 0
    for ep in episodes:
        s = np.asarray(ep["support_indices"]); q = np.asarray(ep["query_indices"])
        classes = np.unique(video[s])
        sc = np.full((len(q), len(classes)), -1e9)
        for ci, c in enumerate(classes):
            sc[:, ci] = (emb[q] @ emb[s[video[s] == c]].T).max(1)
        pred = classes[sc.argmax(1)]
        corr += (pred == video[q]).sum(); tot += len(q)
    return round(100 * corr / tot, 2)


def closed_set(emb, video, cond, idx, rng, cross_mode=False):
    """30-way identification: per video, split captures into gallery/query; classify query among all
    test videos by max cos to gallery. cross_mode: gallery from mode!=query's condition proxy (bw)."""
    vids = np.unique(video[idx])
    gal, qry = [], []
    for v in vids:
        vi = idx[video[idx] == v]
        perm = rng.permutation(vi)
        h = len(perm) // 2
        gal += perm[:h].tolist(); qry += perm[h:].tolist()
    gal = np.array(gal); qry = np.array(qry)
    corr = 0
    for q in qry:
        if cross_mode:
            cand = gal[cond[gal] != cond[q]]
            if len(cand) == 0: cand = gal
        else:
            cand = gal
        sims = emb[q] @ emb[cand].T
        pred = video[cand[sims.argmax()]]
        corr += int(pred == video[q])
    return round(100 * corr / len(qry), 2)


def open_set(emb, video, idx, rng, frac_known=0.5):
    vids = rng.permutation(np.unique(video[idx]))
    nk = int(len(vids) * frac_known)
    known = set(vids[:nk].tolist());
    gal = idx[np.isin(video[idx], list(known))]
    # gallery = half of each known video's captures; queries = the rest of knowns + all unknowns
    gcap, qcap = [], []
    for v in np.unique(video[gal]):
        vi = gal[video[gal] == v]; perm = rng.permutation(vi); h = len(perm) // 2
        gcap += perm[:h].tolist(); qcap += perm[h:].tolist()
    unknown_idx = idx[~np.isin(video[idx], list(known))]
    qcap = np.array(qcap + unknown_idx.tolist()); gcap = np.array(gcap)
    is_known = np.isin(video[qcap], list(known)).astype(int)
    score = (emb[qcap] @ emb[gcap].T).max(1)          # max sim to any enrolled capture
    auroc = round(roc_auc_score(is_known, score), 4)
    # OSCR: among known queries, correct id AND accepted; sweep threshold -> report at EER-ish (median unknown score)
    thr = np.median(score[is_known == 0])
    known_q = qcap[is_known == 1]
    correct_accept = 0
    for q in known_q:
        s = emb[q] @ emb[gcap].T
        if s.max() >= thr and video[gcap[s.argmax()]] == video[q]:
            correct_accept += 1
    oscr = round(100 * correct_accept / len(known_q), 2)
    return {"auroc_known_vs_unknown": auroc, "oscr_at_unknown_median_thr": oscr,
            "n_known_videos": nk, "n_unknown_videos": len(vids) - nk}


def clustering(emb, video, idx):
    X = emb[idx]; y = video[idx]; k = len(np.unique(y))
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    return {"k": k, "ARI": round(adjusted_rand_score(y, lab), 4),
            "NMI": round(normalized_mutual_info_score(y, lab), 4)}


def main():
    device = choose_device("cpu")
    scales = parse_scales("100,500,2000")
    data = normalize_from_base(load_dynamic_multiscale(CACHE, PROTO / "run_manifest.csv", scales))
    video = np.asarray(data.video); cond = np.asarray(data.condition)
    test_idx = np.flatnonzero(data.split == "test")
    test_ep = load_episodes(PROTO / "test_episodes.jsonl")
    cross_ep = [e for e in test_ep if e["scenario"] == "cross_actual_mode_2shot"]
    mixed_ep = [e for e in test_ep if e["scenario"] == "random_mixed_5shot"]

    result = {"n_test_identities": int(len(np.unique(video[test_idx]))), "seeds": {}}
    for seed, ck in CKPTS.items():
        emb = extract_standard(load_swap(Path(ck), scales, device), data,
                               np.arange(len(video)), scales, device, 128, 0)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        result["seeds"][seed] = {
            "fewshot_cross_mode": fewshot(emb, video, cross_ep),
            "fewshot_mixed": fewshot(emb, video, mixed_ep),
            "closed_set_30way": closed_set(emb, video, cond, test_idx, np.random.RandomState(1)),
            "closed_set_30way_cross_mode": closed_set(emb, video, cond, test_idx, np.random.RandomState(1), True),
            "open_set": open_set(emb, video, test_idx, np.random.RandomState(2)),
            "clustering": clustering(emb, video, test_idx),
        }
        print(f"[seed {seed}]", json.dumps(result["seeds"][seed], indent=2))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""YDMS reconstruction, step 3: build a MultiScaleTraffic object + identity split + few-shot episodes,
so the LongEnough AMP training/eval pipeline can be reused unchanged (import only).

Identity split (only identities with >= MIN_SAMPLES samples at D=60): base / validation / test.
Condition = obs_min_bw group (bottleneck), int-coded. abr_mode is set to the condition (a placeholder;
IdentityBatchSampler uses it only when mode_balanced=True, which we keep False). Episodes:
  - random_mixed : support+query from any condition (standard few-shot; also used for subgroup
                   stability by stratifying each query's condition afterwards)
  - cross_bw     : support all in group A, query all in group B, restricted to shared identities
                   (label-intersection) -- the direct cross-mode analog.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent
for _d in (_SCRIPTS / "data", _SCRIPTS / "experiments"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from dataset_paths import DATA as DATA_ROOT

from longenough_offset0_multiscale import MultiScaleTraffic  # noqa: E402

GROUP_TO_ID = {"low_le_1000": 0, "mid_1001_2500": 1, "high_2501_399999": 2, "unlimited_400000": 3}
ID_TO_GROUP = {v: k for k, v in GROUP_TO_ID.items()}
DATA = DATA_ROOT / "ydms_external_v1"


def build_ydms_data(views_npz: Path, conditions_csv: Path,
                    min_samples: int = 11, split=(96, 38, 58), split_seed: int = 20260816):
    z = np.load(views_npz, allow_pickle=False)
    scales = [int(s) for s in z["scales"]]
    arrays = {s: z[f"view_{s}"] for s in scales}
    video = z["label"].astype(np.int64)              # identity id
    run_id = z["run_id"]
    csid = z["canonical_sample_id"].astype(np.int64)
    N = len(video)

    cond_group = {int(r["canonical_sample_id"]): r["obs_min_group"]
                  for r in csv.DictReader(open(conditions_csv))}
    condition = np.array([GROUP_TO_ID.get(cond_group.get(int(c), "other"), -1) for c in csid],
                         dtype=np.int64)

    # identities with enough samples
    ids, counts = np.unique(video, return_counts=True)
    eligible = sorted(int(i) for i, c in zip(ids, counts) if c >= min_samples)
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(eligible)
    nb, nv, nt = split
    base_ids = set(perm[:nb].tolist())
    val_ids = set(perm[nb:nb + nv].tolist())
    test_ids = set(perm[nb + nv:nb + nv + nt].tolist())

    split_arr = np.array(
        ["base" if v in base_ids else "validation" if v in val_ids
         else "test" if v in test_ids else "unused" for v in video])

    # per-identity running repeat index (informational)
    repeat = np.zeros(N, dtype=np.int64)
    seen: dict[int, int] = {}
    for i, v in enumerate(video):
        repeat[i] = seen.get(int(v), 0); seen[int(v)] = repeat[i] + 1

    data = MultiScaleTraffic(
        arrays={s: arrays[s].astype(np.float32) for s in scales},
        sample_index=np.arange(N, dtype=np.int64),
        video=video, split=split_arr, condition=condition,
        repeat=repeat, abr_mode=condition.copy(), run_id=run_id)
    info = {"n_samples": N, "eligible_identities": len(eligible),
            "base": len(base_ids), "validation": len(val_ids), "test": len(test_ids),
            "split_seed": split_seed, "min_samples": min_samples}
    return data, info, {"base": base_ids, "validation": val_ids, "test": test_ids}


def _episodes_for(video, condition, idx_pool, rng, n_way, k_shot, q_query, n_episodes,
                  support_group=None, query_group=None):
    """Generate episodes over identities present in idx_pool. If support/query_group set, draw
    support only from support_group and query only from query_group (label-intersection)."""
    by_id = {}
    for i in idx_pool:
        by_id.setdefault(int(video[i]), []).append(int(i))
    eps = []
    for _ in range(n_episodes):
        if support_group is None:
            usable = [v for v, s in by_id.items() if len(s) >= k_shot + q_query]
        else:
            usable = [v for v, s in by_id.items()
                      if sum(condition[j] == support_group for j in s) >= k_shot
                      and sum(condition[j] == query_group for j in s) >= q_query]
        if len(usable) < n_way:
            continue
        classes = rng.choice(usable, size=n_way, replace=False)
        sup, qry = [], []
        for c in classes:
            pool = np.array(by_id[int(c)])
            if support_group is None:
                sel = rng.permutation(pool)
                sup += sel[:k_shot].tolist(); qry += sel[k_shot:k_shot + q_query].tolist()
            else:
                sp = pool[condition[pool] == support_group]; qp = pool[condition[pool] == query_group]
                sup += rng.choice(sp, size=k_shot, replace=False).tolist()
                qry += rng.choice(qp, size=q_query, replace=False).tolist()
        eps.append({"classes": [int(c) for c in classes],
                    "support_indices": [int(i) for i in sup],
                    "query_indices": [int(i) for i in qry]})
    return eps


def make_episodes(data, split_set, split="test", n_way=10, k_shot=5, q_query=5,
                  n_random=300, n_cross=200, seed=20260816):
    video = data.video; condition = data.condition
    pool = [i for i in range(len(video)) if data.split[i] == split]
    rng = np.random.RandomState(seed)
    out = {"random_mixed": _episodes_for(video, condition, pool, rng, n_way, k_shot, q_query, n_random)}
    # cross-bandwidth label-intersection pairs (only well-populated ones)
    for a, b in [(0, 1), (0, 2), (1, 2)]:      # low<->mid, low<->high, mid<->high
        out[f"cross_{ID_TO_GROUP[a]}_to_{ID_TO_GROUP[b]}"] = _episodes_for(
            video, condition, pool, rng, min(n_way, 6), k_shot, q_query, n_cross, a, b)
        out[f"cross_{ID_TO_GROUP[b]}_to_{ID_TO_GROUP[a]}"] = _episodes_for(
            video, condition, pool, rng, min(n_way, 6), k_shot, q_query, n_cross, b, a)
    return out


if __name__ == "__main__":
    data, info, split_set = build_ydms_data(
        DATA / "ydms_views_60s_v1.npz", DATA / "ydms_conditions_60s_v1.csv")
    print("[build] info:", json.dumps(info))
    from collections import Counter
    print("[build] split sample counts:", dict(Counter(data.split)))
    print("[build] condition counts:", dict(Counter(int(c) for c in data.condition)))
    eps = make_episodes(data, split_set, split="test")
    for k, v in eps.items():
        print(f"  episodes[{k}]: {len(v)}")


def cross_episodes(video, tbin, pool, a, b, rng, n_way=6, k=5, q=5, n=500):
    by = {}
    for i in pool: by.setdefault(int(video[i]), []).append(int(i))
    eps = []
    for _ in range(n):
        usable = [v for v, s in by.items()
                  if sum(tbin[j] == a for j in s) >= k and sum(tbin[j] == b for j in s) >= q]
        if len(usable) < n_way:
            continue
        cls = rng.choice(usable, size=n_way, replace=False)
        sup, qry = [], []
        for c in cls:
            pl = np.array(by[int(c)])
            sup += rng.choice(pl[tbin[pl] == a], size=k, replace=False).tolist()
            qry += rng.choice(pl[tbin[pl] == b], size=q, replace=False).tolist()
        eps.append({"support_indices": sup, "query_indices": qry})
    return eps

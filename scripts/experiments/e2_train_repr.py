#!/usr/bin/env python3
"""Train AMP with the representation levers targeting the ABR information-asymmetry
finding, and measure the cross-mode performance change (esp. on low-bandwidth bw1 queries).

Reuses the AMP model, loss, sampler and episode evaluation unchanged. The only additions are
two optional, physically-motivated representation levers:

  --startup-strip-frac F   zero the first F fraction of fine bins of EVERY run before training
                           (removes the initial pre-buffer burst, which carries little content
                           identity and differs wildly across bandwidth).
  --span-trunc-prob P       during training, with prob P per sample, zero all bins beyond a random
  --span-trunc-min M        cutoff fraction c~U[M,1] (simulate an information-starved / low-bandwidth
                           partial observation of the SAME identity -> teaches completeness/span
                           invariance).

Baseline mode (both levers off) is the control: it must land near the frozen AMP cross-mode test
accuracy (~86 at this script's default capacity, d_model 64 / embedding 96), which confirms the
levers are the only thing changing before any variant is trusted. The headline configuration in
the paper is larger (--d-model 256 --embedding-dim 512) and scores higher.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_SCRIPTS = Path(__file__).resolve().parent.parent
for _d in (_SCRIPTS / "data", _SCRIPTS / "experiments"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from dataset_paths import LONGENOUGH_CACHE, DATA as DATA_ROOT

from longenough_offset0_multiscale import (
    IdentityBatchSampler, LongEnoughMultiScaleDataset,
    load_dynamic_multiscale, normalize_from_base, parse_scales,
)
from train_longenough_offset0_multiscale_baseline import augment, batch_inputs, set_seed, choose_device
from train_longenough_offset0_swapcanonical import (
    AMPEncoder, supervised_contrastive_loss, evaluate_frozen, summarize,
    selection_score, extract_embeddings, load_episodes,
)

CACHE = LONGENOUGH_CACHE
PROTO = DATA_ROOT / "longenough_offset0_protocol_v1"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, default=20260814)
    p.add_argument("--device", default="auto")
    p.add_argument("--startup-strip-frac", type=float, default=0.0)
    p.add_argument("--span-trunc-prob", type=float, default=0.0)
    p.add_argument("--span-trunc-min", type=float, default=0.3)
    p.add_argument("--trunc-mode", choices=("wallclock", "content"), default="wallclock")
    p.add_argument("--aug-kind", choices=("truncate", "randommask"), default="truncate",
                   help="truncate=zero the TAIL (partial observation); randommask=zero the SAME "
                        "number of bins at RANDOM positions (generic-augmentation control)")
    p.add_argument("--consist-weight", type=float, default=0.0,
                   help="explicit full-vs-truncated completeness-invariance loss weight (E3)")
    p.add_argument("--consist-min", type=float, default=0.3)
    # AMP frozen config
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--steps-per-epoch", type=int, default=50)
    p.add_argument("--ids-per-batch", type=int, default=10)
    p.add_argument("--runs-per-id", type=int, default=4)
    p.add_argument("--token-shuffle", choices=("none", "independent", "shared", "rngonly"),
                   default="none",
                   help="ablate the per-position cross-scale correspondence; see "
                        "token_shuffle.py")
    p.add_argument("--encoder", choices=("amp", "globalpool"), default="amp",
                   help="ablation axis: fuse-then-pool (ours) or pool-then-fuse")
    p.add_argument("--loss", choices=("supconce", "triplet"), default="supconce",
                   help="ablation axis: supervised contrastive + CE, or batch-hard triplet")
    p.add_argument("--exclude-condition", type=int, default=-1,
                   help="hold a bandwidth condition out of encoder training AND out of "
                        "the normalization fit (0=bw1 1=bw2 2=bw4 3=bw8); -1 = keep all")
    p.add_argument("--jitter-scale", type=float, default=0.0,
                   help="condition jitter: per-sample scale shift std "
                        "(see cond_jitter.py)")
    p.add_argument("--jitter-dilate", type=float, default=0.0,
                   help="condition jitter: time-dilation range")
    p.add_argument("--normalize", choices=("base", "persample", "both"), default="base",
                   help="input scaling; see persample_norm.py")
    p.add_argument("--tokens", type=int, default=30)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--style-dim", type=int, default=16)
    p.add_argument("--embedding-dim", type=int, default=96)
    p.add_argument("--transformer-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--ce-weight", type=float, default=0.25)
    p.add_argument("--time-drop", type=float, default=0.03)
    p.add_argument("--feature-drop", type=float, default=0.01)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--eval-every", type=int, default=2)
    return p.parse_args()


def strip_startup(arrays, frac):
    if frac <= 0:
        return
    for scale, a in arrays.items():
        k = int(round(frac * a.shape[1]))
        if k > 0:
            a[:, :k, :] = 0.0


def span_truncate(inputs, prob, min_frac, rng, mode="wallclock", sidx=None, cumfrac=None,
                  aug_kind="truncate"):
    """Zero tail bins beyond a random cutoff for a random subset of samples (partial-observation).

    mode=wallclock: cutoff is a random FRACTION of the wall-clock window.
    mode=content:   cutoff is the wall-clock position where cumulative downstream BYTES reach a
                    random budget fraction (physically = a lower-bandwidth trace that observed
                    fewer bytes). Requires sidx (per-sample run index) and cumfrac[N, Fbins].
    """
    if prob <= 0:
        return inputs
    any_scale = next(iter(inputs.values()))
    B = any_scale.shape[0]
    pick = rng.random(B) < prob
    if not pick.any():
        return inputs
    budgets = rng.uniform(min_frac, 1.0, size=B)
    # per-sample time-fraction cutoff
    cf = np.empty(B)
    if mode == "content":
        F = cumfrac.shape[1]
        for i in range(B):
            row = cumfrac[int(sidx[i])]
            j = int(np.searchsorted(row, budgets[i], side="left"))
            cf[i] = min(j + 1, F) / F
    else:
        cf = budgets
    out = {}
    for scale, x in inputs.items():
        x = x.clone()
        nb = x.shape[1]
        for i in range(B):
            if pick[i]:
                c = int(round(cf[i] * nb))
                if aug_kind == "randommask":
                    # zero the SAME number of bins (nb-c) at RANDOM positions (generic control)
                    k = nb - c
                    if k > 0:
                        drop = rng.choice(nb, size=k, replace=False)
                        x[i, drop, :] = 0.0
                else:
                    x[i, c:, :] = 0.0        # zero the TAIL (partial observation)
        out[scale] = x
    return out


def crossmode_perquery(emb, video, cond, episodes, rule="support_max"):
    """Cross-mode accuracy overall and for bw1 (condition==0) queries, under a matching rule."""
    corr = tot = 0
    corr1 = tot1 = 0
    for ep in episodes:
        s = np.asarray(ep["support_indices"]); q = np.asarray(ep["query_indices"])
        classes = np.unique(video[s]); zq = emb[q]
        cols = []
        for c in classes:
            zs = emb[s[video[s] == c]]
            if rule == "support_max":
                cols.append(np.max(zq @ zs.T, axis=1))
            else:  # prototype
                p = zs.mean(axis=0); p /= (np.linalg.norm(p) + 1e-12)
                cols.append(zq @ p)
        scores = np.column_stack(cols)
        pred = classes[np.argmax(scores, axis=1)]; truth = video[q]
        ok = (pred == truth)
        corr += ok.sum(); tot += len(ok)
        m = cond[q] == 0
        corr1 += ok[m].sum(); tot1 += m.sum()
    return corr / tot * 100, (corr1 / tot1 * 100 if tot1 else float("nan"))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    scales = parse_scales("100,500,2000")
    out_dir = DATA_ROOT / "abr_essence_explore_v1" / f"e2_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    class_split = json.loads((PROTO / "class_split.json").read_text())
    base_classes = [int(v) for v in class_split["base_classes"]]
    raw = load_dynamic_multiscale(CACHE, PROTO / "run_manifest.csv", scales)
    strip_startup(raw.arrays, args.startup_strip_frac)   # lever applied on raw bins, pre-normalize
    finest = min(scales)
    down_by_bin = raw.arrays[finest][:, :, 0]            # raw downstream bytes per fine bin
    cum = np.cumsum(down_by_bin, axis=1)
    cumfrac = (cum / np.maximum(cum[:, -1:], 1e-9)).astype(np.float64)  # [N, Fbins]
    from persample_norm import apply_normalization
    from cond_jitter import condition_jitter
    if args.exclude_condition >= 0:
        drop = (raw.split == "base") & (np.asarray(raw.condition) == args.exclude_condition)
        raw.split = np.where(drop, "excluded", raw.split)
        print(f"[HOLDOUT] condition {args.exclude_condition}: "
              f"{int(drop.sum())} base sessions excluded from training and normalization",
              flush=True)
    data, in_dim = apply_normalization(raw, args.normalize, normalize_from_base)
    video = np.asarray(data.video); cond = np.asarray(data.condition)

    base_idx = np.flatnonzero(data.split == "base")
    val_idx = np.flatnonzero(data.split == "validation")
    test_idx = np.flatnonzero(data.split == "test")
    label_map = {v: i for i, v in enumerate(base_classes)}
    dataset = LongEnoughMultiScaleDataset(data, base_idx, label_map)
    sampler = IdentityBatchSampler(dataset, ids_per_batch=args.ids_per_batch,
                                   runs_per_id=args.runs_per_id, steps=args.steps_per_epoch,
                                   seed=args.seed, mode_balanced=False)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    val_ep = load_episodes(PROTO / "validation_episodes.jsonl")
    test_ep = load_episodes(PROTO / "test_episodes.jsonl")
    cross_test = [e for e in test_ep if e["scenario"] == "cross_actual_mode_2shot"]

    from encoder_adapter import GlobalPoolAdapter, batch_hard_triplet
    if args.encoder == "globalpool":
        model = GlobalPoolAdapter(scales, in_dim, args.d_model, args.embedding_dim,
                                  len(base_classes), args.dropout).to(device)
    else:
        kw = dict(scales=scales, in_dim=in_dim, tokens=args.tokens, d_model=args.d_model,
                  style_dim=args.style_dim, embedding_dim=args.embedding_dim,
                  transformer_layers=args.transformer_layers,
                  num_base_classes=len(base_classes), dropout=args.dropout,
                  content_input="observed", temporal_encoder="identity")
        if args.token_shuffle == "none":
            model = AMPEncoder(**kw).to(device)
        else:
            from token_shuffle import TokenShuffledAMP
            model = TokenShuffledAMP(shuffle=args.token_shuffle, **kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * args.steps_per_epoch))
    aug_rng = np.random.RandomState(args.seed + 7)

    best_score, best_epoch = -math.inf, -1
    ckpt = out_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch); model.train(); tot_loss = 0.0
        for batch in loader:
            inputs = batch_inputs(batch, scales, device)
            inputs = augment(inputs, args.time_drop, args.feature_drop, args.noise_std)
            inputs = condition_jitter(inputs, args.jitter_scale, args.jitter_dilate)
            sidx = batch["sample_index"].numpy()
            inputs = span_truncate(inputs, args.span_trunc_prob, args.span_trunc_min, aug_rng,
                                   args.trunc_mode, sidx, cumfrac, args.aug_kind)
            labels = batch["label"].to(device)
            output = model(inputs)
            if args.loss == "triplet":
                loss = batch_hard_triplet(output["embedding"], labels)
            else:
                loss = supervised_contrastive_loss(output["embedding"], labels, args.temperature) \
                    + args.ce_weight * F.cross_entropy(output["identity_logits"], labels)
            if args.consist_weight > 0:
                trunc_inputs = span_truncate(inputs, 1.0, args.consist_min, aug_rng,
                                             args.trunc_mode, sidx, cumfrac, args.aug_kind)
                z_trunc = model(trunc_inputs)["embedding"]
                consist = (1.0 - F.cosine_similarity(output["embedding"], z_trunc, dim=1)).mean()
                loss = loss + args.consist_weight * consist
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot_loss += float(loss.detach().cpu())
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            emb = extract_embeddings(model, data, val_idx, scales, device, 128, 0)
            score = selection_score(summarize(evaluate_frozen(emb, data.video, val_ep)))
            if score > best_score:
                best_score, best_epoch = score, epoch
                torch.save({"model_state": model.state_dict(), "base_classes": base_classes,
                            "args": {"content_input": "observed", "temporal_encoder": "identity",
                                     "tokens": args.tokens, "d_model": args.d_model,
                                     "style_dim": args.style_dim, "embedding_dim": args.embedding_dim,
                                     "transformer_layers": args.transformer_layers, "dropout": args.dropout}},
                           ckpt)
            print(f"[E{epoch:03d}] loss={tot_loss/args.steps_per_epoch:.4f} val_joint={score:.4f}", flush=True)
        else:
            print(f"[E{epoch:03d}] loss={tot_loss/args.steps_per_epoch:.4f}", flush=True)

    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    sel = np.concatenate((val_idx, test_idx))
    emb = extract_embeddings(model, data, sel, scales, device, 128, 0)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    cross_val = [e for e in val_ep if e["scenario"] == "cross_actual_mode_2shot"]
    val_sm, _ = crossmode_perquery(emb, video, cond, cross_val, "support_max")
    val_pr, _ = crossmode_perquery(emb, video, cond, cross_val, "prototype")
    cross_all, cross_bw1 = crossmode_perquery(emb, video, cond, cross_test, "support_max")
    cross_all_pr, cross_bw1_pr = crossmode_perquery(emb, video, cond, cross_test, "prototype")
    final_summary = summarize(evaluate_frozen(emb, data.video, val_ep + test_ep))
    result = {
        "tag": args.tag, "seed": args.seed, "best_epoch": best_epoch,
        "startup_strip_frac": args.startup_strip_frac,
        "span_trunc_prob": args.span_trunc_prob, "span_trunc_min": args.span_trunc_min,
        "trunc_mode": args.trunc_mode,
        "consist_weight": args.consist_weight, "consist_min": args.consist_min,
        "cross_val_supportmax": round(val_sm, 2), "cross_val_prototype": round(val_pr, 2),
        "cross_mode_test_supportmax": round(cross_all, 2),
        "cross_mode_test_prototype": round(cross_all_pr, 2),
        "cross_mode_test_bw1_query": round(cross_bw1, 2),
        "cross_mode_test_bw1_query_prototype": round(cross_bw1_pr, 2),
        "scenarios_test": {r["scenario"]: round(r["mean_accuracy"] * 100, 2)
                           for r in final_summary if r["split"] == "test" and r["method"] == "support_max"},
    }
    print(json.dumps(result, indent=2))
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    # per-query correctness (support_max) for identity-clustered significance
    import csv as _csv
    with (out_dir / "perquery.csv").open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["seed", "model", "scenario", "split", "offset", "episode", "query_pos",
                    "query_identity", "correct"])
        for ep in cross_test:
            s = np.asarray(ep["support_indices"]); q = np.asarray(ep["query_indices"])
            classes = np.unique(video[s]); zq = emb[q]
            scores = np.column_stack([np.max(zq @ emb[s[video[s] == c]].T, axis=1) for c in classes])
            pred = classes[np.argmax(scores, axis=1)]; truth = video[q]
            for pos in range(len(q)):
                w.writerow([args.seed, args.tag, "cross_actual_mode_2shot", "test", 0,
                            int(ep["episode"]), pos, int(truth[pos]), int(pred[pos] == truth[pos])])


if __name__ == "__main__":
    main()

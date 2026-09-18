# PEA-VR

Official implementation of **PEA-VR** (Partial-Evidence Alignment for Video Recognition) — recognizing
encrypted video from traffic alone, few-shot, robust across ABR bitrate regimes.

Paper: *PEA-VR: Few-Shot Encrypted Video Recognition via Partial-Evidence Alignment under Adaptive
Streaming* (Computer Networks).

This repository holds the code for the method the paper proposes: data preparation, training, and
the evaluations behind its reported numbers. The manuscript, datasets, model checkpoints and run
artifacts are not part of it.

## Layout

```
scripts/
  data/          dataset paths, model inputs, frozen protocol (6 files)
  experiments/   encoders, training harness, evaluation (10 files)
datasets/        where the scripts look for the data (not tracked; see Setup)
runs/            created on first run; where outputs are written (not tracked)
```

## Setup

```bash
pip install -r requirements.txt
```

Neither dataset ships here. Place or symlink them under `datasets/` at the repository root, which
git ignores:

```
datasets/LongEnough_offset0_60s_packet_cache_v1/     # LongEnough, offset-0 packet cache
datasets/mobile_yt_dataset/                          # YDMS, only for the YDMS input builders
```

To keep them somewhere else, set the matching environment variable instead:

```bash
export PEA_VR_LONGENOUGH_CACHE=/elsewhere/LongEnough_offset0_60s_packet_cache_v1
export PEA_VR_YDMS_ROOT=/elsewhere/mobile_yt_dataset
export PEA_VR_DATA=/elsewhere/runs      # optional; output root, default `runs/`
```

Everything resolves in `scripts/data/dataset_paths.py`, the only place a path is defined. Paths are
anchored at the repository root, so the scripts behave the same from any working directory. Scripts
that also take a path on the command line let that argument win.

## Running

Run from the repository root.

**Step 0 — the frozen protocol.** Every experiment reads the class-disjoint splits and episode lists
from `$PEA_VR_DATA/longenough_offset0_protocol_v1/`. `prepare_longenough_offset0_protocol.py` is the
script that froze them, but it does not *create* the class split: it normalizes an existing
`class_split.json` and records hashes, and the source split it reads
(`longenough_smoke_v1/class_split.json`) is not part of this repository. Obtain the frozen protocol
directory together with the dataset; re-running the freeze from scratch is not supported here.

```bash
# 1. train PEA-VR (writes result.json, perquery.csv and best.pt per run)
python scripts/experiments/e2_train_repr.py \
    --d-model 256 --embedding-dim 512 --span-trunc-prob 0.5 --tag trunc

# 2. evaluation, inference-only; each needs checkpoints from step 1
python scripts/experiments/crossbw_matrix.py
python scripts/experiments/unseen_bandwidth_eval.py
python scripts/experiments/multiprotocol_eval.py
```

Ablation arms come from the same harness as step 1: `--span-trunc-prob 0` for no tail truncation,
`--encoder globalpool` for the global-pooling arm.

The evaluation scripts read checkpoints from fixed run-directory names under `$PEA_VR_DATA`
(`e2_cap_d256e512_s{814,815,816}`, `e2_ho{0,3}_d256e512_s{814,815,816}`,
`longenough_offset0_full100_alignedpool_seed{1,2}_v1`). Tag your training runs accordingly, or edit
the constants at the top of each script.

## What is here

**`scripts/data/`**

| Script | Role |
|---|---|
| `dataset_paths.py` | the single place dataset and output locations are resolved |
| `longenough_offset0_multiscale.py` | multiscale views (100/500/2000 ms, 5 features per bin) over the LongEnough packet cache; also the shared dataset, normalization and identity sampler |
| `prepare_longenough_offset0_protocol.py` | freeze the class-disjoint splits and the few-shot episodes |
| `build_ydms_packet_npz.py` | YDMS step 1: lossless packet-level NPZ from the raw captures |
| `build_ydms_views.py` | YDMS step 2: the same multiscale views, so one encoder consumes both datasets |
| `ydms_protocol.py` | YDMS step 3: identity split and episode construction |

The YDMS side keeps only this input-construction path. The YDMS training and evaluation scripts are
not included; what remains is enough to build YDMS into the view format the model consumes.

**`scripts/experiments/`**

| Group | Scripts |
|---|---|
| Encoders | `train_longenough_offset0_swapcanonical.py` (`AMPEncoder`), `train_longenough_offset0_multiscale_baseline.py` (global pooling, plus utilities every script imports), `encoder_adapter.py` |
| Training harness | `e2_train_repr.py` — trains PEA-VR and every ablation arm; `cond_jitter.py`, `persample_norm.py`, `token_shuffle.py` are its optional input/augmentation levers |
| Bandwidth generalization | `crossbw_matrix.py` (reference x query retrieval matrix), `unseen_bandwidth_eval.py` (episodes drawn entirely from a held-out condition) |
| Protocol transfer | `multiprotocol_eval.py` — few-shot / closed-set / open-set / clustering on one frozen embedding |

### Naming

Docstrings and class names follow the paper. A few file names and flags keep their original
spelling, because run directories and saved tags depend on them:

| In the paper | In the code |
|---|---|
| AMP encoder | `AMPEncoder`, defined in `train_longenough_offset0_swapcanonical.py` |
| global pooling (ablation arm) | `MultiScaleTemporalEncoder`, in `train_longenough_offset0_multiscale_baseline.py` |
| temporal tail truncation | the `--span-trunc-prob` / `--span-trunc-min` flags and `span_truncate` |

`*_baseline.py` refers to the **global-pooling ablation arm of the proposed method**.

## Datasets

| Dataset | Source |
|---|---|
| LongEnough | Hasselquist et al., "Raising the Bar: Improved Fingerprinting Attacks and Defenses for Video Streaming Traffic", *PoPETs* 2024(4):167-184. [doi:10.56553/popets-2024-0112](https://doi.org/10.56553/popets-2024-0112)<br>Variable-bandwidth collection extended by Carlson et al., "Understanding and Improving Video Fingerprinting Attack Accuracy under Challenging Conditions", *WPES* 2024:141-154. [doi:10.1145/3689943.3695045](https://doi.org/10.1145/3689943.3695045) |
| YDMS | Loh et al., "YouTube Dataset on Mobile Streaming for Internet Traffic Modeling and Streaming Analysis", *Scientific Data* 9:293 (2022). [doi:10.1038/s41597-022-01418-y](https://doi.org/10.1038/s41597-022-01418-y) |

We use the undefended offset-0 subset of LongEnough's extended variable-bandwidth collection:
100 video identities x 4 bandwidth presets x 10 sessions = 4,000 sessions, first 60 s of each.

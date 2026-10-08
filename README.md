# PEA-VR

PEA-VR is a complete training and evaluation package for few-shot encrypted video recognition with partial traffic observations. It implements reliability-aware aligned multiscale pooling (AMP), structured traffic fingerprints, Partial Monotone Alignment (PMA), partial-view learning, and compatibility-weighted support aggregation.

## Contents

- [Workflow and repository layout](#workflow-and-repository-layout)
- [Environment installation](#environment-installation)
- [Public dataset acquisition](#public-dataset-acquisition)
- [Dataset construction](#dataset-construction)
- [Fixed evaluation protocols](#fixed-evaluation-protocols)
- [Model and matching pipeline](#model-and-matching-pipeline)
- [Training and checkpoint resume](#training-and-checkpoint-resume)
- [Running the experiment matrix](#running-the-experiment-matrix)
- [Evaluation and diagnostics](#evaluation-and-diagnostics)
- [Reporting and independent metric checks](#reporting-and-independent-metric-checks)
- [New video enrollment and recognition](#new-video-enrollment-and-recognition)
- [Automated tests](#automated-tests)
- [Troubleshooting and reproducibility](#troubleshooting-and-reproducibility)

## Workflow and repository layout

The execution order is:

```text
Public archives
    -> verified raw packet/player files
    -> filtering, deduplication, and identity splits
    -> training-only normalization and mode clustering
    -> prepared dataset and fixed validation/test protocols
    -> episodic training and validation-selected checkpoints
    -> recognition, truncation, open-set, and alignment evaluation
    -> independent metric checks, tables, and figures
```

The source tree is organized as follows:

```text
PEA-VR/
  pyproject.toml                 Package metadata and CLI entry point
  requirements/                  Reference versions and Windows dependency locks
  experiments/
    matrix.json                  Complete experiment manifest
    configs/*.yaml               Fully specified training configurations
  src/pea_vr/
    cli.py                       Command-line argument parsing and dispatch
    config.py                    Defaults, configuration merging, and validation
    data/
      download.py                Versioned downloads, checksums, and extraction
      remote_zip.py              Selective LongEnough HTTP Range acquisition
      adapters.py                Official packet and player-log formats
      features.py                Binning, observation masks, and normalization
      prepare.py                 Filtering, splits, mode fitting, and dataset loading
      protocols.py               Fixed episodes and open-set partitions
    models/
      amp.py                     Multi-scale encoder and structured fingerprints
      pooling.py                 Deterministic adaptive mean pooling
      pma.py                     PMA and Soft-DTW dynamic programming
      partial_views.py           Prefix masks, progress warps, and alignment targets
      matching.py                Pair scores and candidate aggregation
      baselines.py               Adapted comparison encoders
      factory.py                 Model construction from configuration
    training/
      engine.py                  Training, validation, checkpoints, and adaptation
      losses.py                  Alignment, contrastive, and triplet objectives
    evaluation.py                Recognition and diagnostic measurements
    inference.py                 Packet-only enrollment and recognition
    experiments.py               Experiment matrix generation
    runner.py                    Matrix execution and resume
    selection.py                 Validation-only Soft-DTW smoothing selection
    reporting.py                 Seed aggregation and plots
    tables.py                    Completeness-checked table export
    case_study.py                Correspondence and support-responsibility figures
    audit.py                     Dataset and protocol integrity checks
    utils.py                     Hashes, random states, device setup, and JSON I/O
  scripts/recompute_metrics.py   Independent metric recomputation
  tests/                         Mathematical, data, training, and integration tests
```

Commands below use PowerShell and keep large artifacts on `D:`. Run installation commands and commands with relative repository paths from the project root. After installation, `pea-vr` can run from any working directory when its file arguments are absolute. `python -m pea_vr` is an equivalent entry point.

A typical external workspace is:

```text
D:\PEA-VR-data\raw\
  longenough\
  ydms\
D:\PEA-VR-work\
  prepared\longenough\
  prepared\ydms\
  protocols\longenough\validation\
  protocols\longenough\test\
  protocols\ydms\validation\
  protocols\ydms\test\
  runs\RUN_NAME\
  summary\
  tables\
  figures\
```

Keep each dataset version and each training configuration in its own output directory. Raw archives, prepared data, and trained checkpoints are generated artifacts and are not bundled with the source package.

## Environment installation

### Supported runtime

Use Python **3.10, 3.11, or 3.12**. The reference runtime uses Python 3.12, PyTorch 2.6.0, NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.5.1, PyYAML 6.0.1, requests 2.32.3, pytest 7.4.4, and matplotlib 3.9.2.

CUDA accelerates training and PMA evaluation. CPU execution is supported for tests, data preparation, training, and inference. Full-budget experiment runtime depends on the device and method; source installation does not start training automatically.

### Windows, Python 3.12, CUDA 11.8 wheels

From the project root:

```powershell
python -m venv D:\PEA-VR-env
D:\PEA-VR-env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements/windows-py312-cu118.txt
python -m pip install --no-deps -e ".[test,plots]"
python -m pip check
pea-vr --version
pea-vr --help
```

The full Windows lock files include transitive dependencies. The `test` and `plots` extras describe the development and visualization dependencies; those dependencies are already covered by the corresponding lock file.

Check the installed device before starting a CUDA run:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Use `--device cuda` to require CUDA, `--device cpu` to require CPU, or `--device auto` to select CUDA when available. Requesting CUDA when unavailable produces an error.

### Windows, Python 3.12, CPU wheels

Use a separate environment for the CPU dependency set:

```powershell
python -m venv D:\PEA-VR-env-cpu
D:\PEA-VR-env-cpu\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/windows-py312-cpu.txt
python -m pip install --no-deps -e ".[test,plots]"
python -m pip check
```

Activation is optional: invoke `D:\PEA-VR-env\Scripts\python.exe` and `D:\PEA-VR-env\Scripts\pea-vr.exe` directly if preferred.

### Other supported Python/platform combinations

The top-level reference requirements can be used outside the Windows/Python 3.12 lock-file combination. For example, on Linux with CPU wheels:

```bash
python3 -m venv /data/pea-vr-env
source /data/pea-vr-env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/reference.txt
python -m pip install -e ".[test,plots]"
python -m pip check
```

For CUDA, select the `cu118` PyTorch wheel index instead. Replace the Windows paths in the remaining commands with local absolute paths. `requirements/reference.txt` pins the principal packages; it is not a complete transitive dependency lock. Each training run saves the numerical runtime and actual device in `environment.json`.

## Public dataset acquisition

### YDMS

```powershell
pea-vr download ydms --output D:\PEA-VR-data\raw\ydms
```

The downloader reads the fixed **Figshare article version 2**, downloads its files, verifies the published MD5 values, records local SHA-256 values, and extracts ZIP files under `raw\ydms\extracted`. Data provenance and completed-file records are saved as `source_record.json` and `download_manifest.json`.

Downloads use bounded HTTP Range requests and a `.part` file. Rerun the same command after a network interruption to continue the download. Extraction requires additional space for the uncompressed data.

To separate download and extraction:

```powershell
pea-vr download ydms --output D:\PEA-VR-data\raw\ydms --no-extract
pea-vr extract D:\PEA-VR-data\raw\ydms\mobile_yt_dataset.zip --output D:\PEA-VR-data\raw\ydms\extracted
```

Public source: [YouTube Dataset on Mobile Streaming](https://figshare.com/articles/dataset/19096823).

### LongEnough

```powershell
pea-vr download longenough --output D:\PEA-VR-data\raw\longenough
```

The default acquisition reads `LongEnough-variable.zip` and `LongEnough-variable-extended.zip` from the public author directory. It fetches ZIP members through verified HTTP Range reads, selecting **offset 0000**, all video identities, and all repeats. The primary cohort comprises 100 identities, bandwidth factors **1, 2, 4, and 8**, and 10 repeats per identity/bandwidth: **4,000 sessions**.

For each session, acquisition keeps the traffic `.log`, player `.qoe.log`, and bandwidth `.bw` files. ZIP member CRC values are checked during extraction; the local manifest records member sizes, CRC values, archive names, and SHA-256 values. Reruns verify completed members before reusing them. This workflow fetches the requested members without expanding the full two archives.

`--videos` and `--offsets` select other acquisition subsets. A subset still has to satisfy the dataset and episode requirements chosen later; downloading fewer identities does not change the primary protocol.

Alternatively, download and extract the archives from the [LongEnough public directory](https://liuonline-my.sharepoint.com/:f:/g/personal/davha914_student_liu_se/ErK6esYd5IdOiuvfLnXK6NoBEdlj579MlXBvG2wkfQEozg?e=sCHtWp). Dataset construction recursively discovers the official files under the supplied source directory.

## Dataset construction

### Build the primary datasets

```powershell
pea-vr prepare longenough `
  --source D:\PEA-VR-data\raw\longenough `
  --output D:\PEA-VR-work\prepared\longenough

pea-vr prepare ydms `
  --source D:\PEA-VR-data\raw\ydms `
  --output D:\PEA-VR-work\prepared\ydms
```

Each `prepare` command performs the following stages in order.

### 1. Discover and parse paired records

| Dataset | Traffic input | Player/metadata input | Identity and direction |
|---|---|---|---|
| LongEnough | `VVVV-OOOO-RRRR.log`: relative nanoseconds, `s/r`, on-wire packet length, epoch milliseconds, and TCP metadata | Matching `.qoe.log`; `.bw` contains the bandwidth `scale` | Video identity comes from `VVVV`; sent packets are uplink and received packets are downlink |
| YDMS | `run_*/video_traffic.csv`: `timestamp`, `ipSrc`, `ipDst`, `tcpLen`, `udpLen`, `payloadProtocolNumber` | Matching `application_data.csv` | `videoid` must identify one video; the private client endpoint determines packet direction |

YDMS retains TCP/UDP records and uses the dataset's native TCP/UDP length fields without adding synthetic packet headers. LongEnough uses its on-wire lengths. Packet epochs synchronize player observations; the recognition input uses relative packet time.

The observation window starts at the first selected packet and covers **[0, 60) seconds**, including startup. Sessions with less than 60 seconds of traffic are excluded. Packet caches retain the first 120 seconds, when available, for diagnostics that inspect later windows. Full capture duration is recorded separately.

LongEnough quality trajectories use player bitrate events. YDMS converts `fmt` using a fixed itag-to-height table and converts `bh` from milliseconds to seconds. Before the first YDMS player event, startup uses quality 0, buffer 0, and stalled state 1. A session without player observations inside the 60-second traffic window is excluded. Future events are not copied backward into that startup interval.

### 2. Remove duplicates before splitting identities

The first 60 seconds of packet times, directions, and lengths produce signatures with 1 microsecond and 1 millisecond time quantization. Matching signatures form duplicate groups before train/validation/test splitting. One representative is retained for a same-identity group; all records in a group containing conflicting identities are excluded.

YDMS identities must retain at least **11 usable sessions** after filtering and deduplication. Exclusions and final cohort counts are written before strict cohort checks, so a count mismatch can be investigated from the saved records.

### 3. Create identity-disjoint splits

| Dataset | Training identities | Validation identities | Test identities | Split seed |
|---|---:|---:|---:|---:|
| LongEnough | 50 | 20 | 30 | 20260814 |
| YDMS | 96 | 38 | 58 | 20260816 |

The primary LongEnough build also checks exactly 4,000 sessions and exactly 10 repeats, numbered 0 through 9, for each identity at each bandwidth. Identity sets are disjoint across the three splits.

Custom cohorts can use `--allow-cohort-difference --counts TRAIN VALIDATION TEST`. The metadata records that the cohort is non-primary. This option changes the requested identity counts; it does not create records or relax episode eligibility. Primary experiments use the default strict build.

### 4. Compute packet features and observation masks

Each scale is computed independently from the same packet trace. The feature order is:

| Channel | Statistic in one time bin |
|---:|---|
| 0 | Downlink bytes |
| 1 | Uplink bytes |
| 2 | Downlink packet count |
| 3 | Uplink packet count |
| 4 | Downlink bytes divided by downlink packet count; zero if the count is zero |

The default scales are **100, 500, and 2,000 ms**, giving **600, 120, and 30 bins** over 60 seconds. Before adding the mask channel, a single session has feature shapes `(600, 5)`, `(120, 5)`, and `(30, 5)`.

Each scale also has a separate observation mask. An observed bin with no packets is a valid zero-traffic bin; an unavailable bin has mask 0. For an observation cutoff, bin-center times determine mask availability, while packets are selected using the half-open observation interval. All positive scales must divide the 60-second window and be multiples of the finest scale.

### 5. Fit and freeze training-only transformations

Packet normalization applies `log1p` to all five nonnegative statistics and fits one mean and standard deviation per feature at each scale, using observed bins from training sessions only. Observed zero-traffic bins participate in fitting. Near-zero standard deviations are replaced by 1; unavailable bins are set to zero after transformation.

Mode construction uses separate player-derived feature vectors:

- **LongEnough:** a 30-point quality trajectory sampled every 2 seconds, plus summary statistics. Three clusters are named L, T, and H by increasing quality. Silhouette scores for K=2 through K=10 are saved as diagnostics; the operational mode model uses K=3.
- **YDMS:** stall ratio, low-buffer ratio, median buffer, tail buffer, player-state ratio, low-quality ratio, mean log quality, and quality-switch ratio. Three clusters are named S, T, and H using their player-state and quality characteristics.

Mode-feature standardization and K-means are fitted on training sessions only. Their frozen parameters assign validation/test modes. In held-out bandwidth experiments, the excluded bandwidth also does not contribute to normalization or mode fitting.

### 6. Persist the prepared dataset

```text
prepared/DATASET/
  dataset.json            Dataset settings, counts, file hashes, and fingerprint
  cohort_inventory.json   Retained identity counts and exclusion totals
  sessions.jsonl          Session IDs, identity, split, mode, condition, cache paths
  excluded.jsonl          Rejected source records and reasons
  split.json              Fixed identity lists
  normalizer.json         Training fit indices and per-scale statistics
  modes.json              Training fit indices, centers, names, and diagnostics
  features.npz            Unnormalized packet statistics and masks
  packets/*.npz           Relative times, directions, and lengths
  players/*.json          Separate player records used by diagnostics
  records/*.json          Verified per-session parsing cache
```

`PreparedDataset` verifies metadata and file hashes, then applies the frozen normalizer in memory. Model batches read packet features or packet caches; they do not read player feature vectors.

An interrupted preparation can reuse verified parsing caches when rerun against the same unfinished output directory. Once `dataset.json` exists, `prepare` refuses to overwrite that completed dataset. Use a new output directory when changing sources, scales, cohort settings, or held-out conditions.

## Fixed evaluation protocols

### Generate validation and test episodes

```powershell
pea-vr protocols --data D:\PEA-VR-work\prepared\longenough --split validation --output D:\PEA-VR-work\protocols\longenough\validation
pea-vr protocols --data D:\PEA-VR-work\prepared\longenough --split test --bandwidth-matrix --output D:\PEA-VR-work\protocols\longenough\test
pea-vr protocols --data D:\PEA-VR-work\prepared\ydms --split validation --output D:\PEA-VR-work\protocols\ydms\validation
pea-vr protocols --data D:\PEA-VR-work\prepared\ydms --split test --output D:\PEA-VR-work\protocols\ydms\test
```

The default protocol seed is **20260817**. Scenario-specific seeds are derived deterministically. Each JSONL episode stores stable support/query session IDs, identity labels, modes, and bandwidth conditions. `protocols.json` records episode counts, file SHA-256 values, and the prepared dataset fingerprint.

In the table below, queries are **per identity**.

| Dataset/scenario | Ways | Shots | Queries | Episodes |
|---|---:|---:|---:|---:|
| LongEnough `cross_mode` | 5 | 2 | 2 | 300: 50 for each of six directed L/T/H pairs |
| LongEnough `same_mode` | 5 | 2 | 2 | 300: 100 per mode |
| LongEnough `mode_balanced` | 5 | 3 | 3 | 300: one support and one query per mode |
| LongEnough `random_mixed` | 5 | 5 | 5 | 300 |
| LongEnough `cross_bandwidth` | 5 | 2 | 2 | 300: 75 each for 1→8, 8→1, 1→2, and 2→1 |
| LongEnough `bandwidth_A_to_B` | 5 | 2 | 2 | 300 for each of 16 diagnostic cells |
| LongEnough `heldout_bw1` / `heldout_bw8` | 5 | 2 | 2 | 300 per held-out cohort |
| YDMS `random_mixed` | 10 | 5 | 5 | 300 |
| YDMS `T_to_H` | 6 | 5 | 5 | 500 |
| YDMS `H_to_T` | 6 | 5 | 5 | 500 |

Support and query sessions are disjoint within each episode. A scenario fails if too few identities satisfy its support/query conditions; ways and shots are not reduced automatically. Reuse the same protocol files for all methods in a comparison.

The primary LongEnough test build additionally writes `open_set.jsonl`: five fixed partitions of 15 known and 15 unknown test identities. Each known identity contributes five gallery supports; its remaining sessions and all unknown sessions become queries. Unknown query labels are -1.

### Audit data and protocols

```powershell
pea-vr audit --data D:\PEA-VR-work\prepared\longenough --protocols D:\PEA-VR-work\protocols\longenough\test --output D:\PEA-VR-work\audit_longenough.json
pea-vr audit --data D:\PEA-VR-work\prepared\ydms --protocols D:\PEA-VR-work\protocols\ydms\test --output D:\PEA-VR-work\audit_ydms.json
```

Auditing checks dataset/cache hashes, identity isolation, duplicate signatures, training-only fit indices, protocol hashes, episode counts, labels, conditions, and support/query separation. Training checkpoint selection requires protocols from the **validation** split; final recognition measurements use the **test** split.

## Model and matching pipeline

The full PEA-VR forward pass has five stages:

1. **Per-scale encoding.** Concatenate the five normalized statistics and the observation mask. Three independent CNN branches use Conv1d 6→128→256, kernel 5, stride 1, GELU, and dropout 0.1. Adaptive mean pooling produces M=30 tokens per scale; pooling the mask produces token reliability.
2. **Scale fusion.** A 256→256 projection and 256→1 gate produce reliability-weighted scale weights. Fused tokens pass through GELU and LayerNorm. Fusion reliability uses the maximum reliability across scales.
3. **Fingerprint construction.** Reliability-weighted mean and valid-token maximum pooling are concatenated and projected 512→1024→512, then L2-normalized to obtain `z`. The structured fingerprint stores `z: (512,)`, `U: (30, 256)`, and `r: (30,)`. A 256→128 normalized local projection supplies matching tokens `V`.
4. **Pair matching.** PMA processes local cosine affinities with `delta=0.2`, `gap=0.1`, and `gamma=0.1`. Its dynamic program includes restart transitions and soft aggregation over endpoints. The resulting correspondence mass produces local similarity and reliability-weighted overlap. Pair similarity mixes global and local evidence with `lambda_z=0.5`.
5. **Candidate aggregation.** Supports of the same identity receive compatibility weights from local similarity and overlap. Candidate scores use weighted log-sum-exp with `tau_c=0.2` and `tau_s=0.1`. A one-support candidate reduces to its pair score.

PMA correspondence mass remains differentiable so the alignment loss can backpropagate through it. The deterministic pooling implementation uses the same adaptive interval boundaries as adaptive average pooling and supports strict deterministic CUDA backward.

Matching alternatives are `diagonal`, `soft_dtw`, and global-only `none`. Aggregation alternatives include mean, support maximum, uniform log-sum-exp, similarity-only, overlap-only, compatibility, and normalized class prototypes. Model-specific configurations select valid combinations.

## Training and checkpoint resume

### Train the full method

```powershell
pea-vr train `
  --config experiments/configs/longenough_pea_vr_20260814.yaml `
  --data D:\PEA-VR-work\prepared\longenough `
  --validation-protocols D:\PEA-VR-work\protocols\longenough\validation `
  --output D:\PEA-VR-work\runs\longenough_pea_vr_20260814 `
  --device cuda
```

For YDMS, use `experiments/configs/ydms_pea_vr_20260814.yaml`, the YDMS prepared dataset, its validation protocols, and a separate `ydms_pea_vr_20260814` run directory.

The fully expanded YAML files are the run specifications. Unknown configuration keys, nonfinite values, incompatible alignment losses, and invalid scales are rejected during loading. CLI `--device` overrides `training.device`.

### One training episode

Each default training episode selects **10 identities**, **2 supports**, and **4 queries per identity**, giving 20 supports and 40 queries. Sampling uses training identities only and excludes a held-out bandwidth when configured.

For full PEA-VR, the engine then:

1. Loads normalized multi-scale packet inputs and computes original fingerprints.
2. Independently applies a prefix mask to each session with probability 0.5; retained fraction is uniform in `[0.3, 1]`. Training prefix augmentation masks the existing feature bins and observation masks by their center times.
3. Applies a monotone three-control-point progress warp with probability 0.5 and `eta` in `[-0.3, 0.3]`. Warping interpolates encoded `U/r` and recomputes `V`; it does not change packet byte/count statistics. The global embedding comes from the prefix view.
4. Scores transformed queries against transformed supports using the same matcher used for inference, and computes episodic cross-entropy.
5. Compares the original and transformed local evidence with a column-normalized Gaussian correspondence target of sigma **2 seconds**.
6. Computes supervised contrastive loss on original/transformed global embeddings.
7. Backpropagates `Lepi + 0.5 * Lali + 0.2 * Lsup`, checks finite gradients, clips the gradient norm, and updates the optimizer and scheduler once.

Query and alignment losses are accumulated in chunks. The defaults are `query_chunk=4`, `alignment_chunk=8`, and matching `pair_chunk=128`; these control memory use without changing the intended episode objective.

### Default optimizer and schedule

| Setting | Value |
|---|---|
| Training budget | 30,000 episodes/optimizer updates |
| Main seeds | 20260814, 20260815, 20260816 |
| Sensitivity seeds | 20260814, 20260815 |
| Optimizer | AdamW |
| Initial / final learning rate | 1e-4 / 1e-6, cosine schedule |
| Weight decay | 1e-4 |
| Adam betas / epsilon | (0.9, 0.999) / 1e-8 |
| Gradient norm limit | 1.0 |
| Episodic / contrastive temperature | 0.1 / 0.1 |
| Validation interval | 500 episodes, and final episode |
| Checkpoint interval | 1,000 episodes, and controlled stopping point |
| Retained periodic checkpoints | Most recent 3, plus separate best/latest |
| Numeric execution | FP32, TF32 disabled, deterministic algorithms enabled |

Validation uses fixed validation-identity episodes: `cross_mode` for primary LongEnough, `random_mixed` for YDMS, or the matching held-out scenario. Validation preserves the training random stream. `best.pt` updates only when validation accuracy improves; ties keep the earlier checkpoint.

### Run outputs

```text
runs/RUN_NAME/
  config.json                Resolved run configuration
  environment.json           Python/package/CUDA versions and actual GPU
  dataset.json               Prepared dataset metadata and fingerprint
  training.jsonl             Episode, losses, learning rate, gradient norm, timing
  validation_XXXXXX.json      Validation measurements at saved evaluation steps
  status.json                Actual completed episode, budget completion, best step
  latest.pt                  Resume checkpoint
  best.pt                    Validation-selected evaluation checkpoint
  episode_XXXXXX.pt          Retained periodic checkpoints
```

Checkpoint payloads include model, optimizer, scheduler, completed step, best validation state, frozen packet normalizer, dataset metadata, configuration/implementation fingerprints, validation protocol manifest, Python/NumPy/PyTorch/CUDA random states, episode sampler state, and view-generator state. Checkpoint files are written through a temporary file before replacement.

### Exact resume

```powershell
pea-vr train `
  --config experiments/configs/longenough_pea_vr_20260814.yaml `
  --data D:\PEA-VR-work\prepared\longenough `
  --validation-protocols D:\PEA-VR-work\protocols\longenough\validation `
  --output D:\PEA-VR-work\runs\longenough_pea_vr_20260814 `
  --resume D:\PEA-VR-work\runs\longenough_pea_vr_20260814\latest.pt `
  --device cuda
```

Exact resume checks the resolved configuration, prepared dataset fingerprint, implementation version, validation protocols, and numerical runtime. Keep the same device setting and runtime; start a separate run when intentionally changing the configuration. `latest.pt` preserves the training state, while `best.pt` selects the model for evaluation.

For a controlled interruption, add `--max-steps 1000` to the initial training command. This stops at **global episode 1000** and writes a checkpoint while retaining the 30,000-episode scheduler and budget. Resume without that option to finish the budget. It is an absolute stop step, not an additional number of updates. A process interruption between saved checkpoints resumes from the last saved step.

### Adapted baseline training

All methods share identity splits and evaluation protocol files, while using their own configurations and training objectives.

| Configuration method | Encoder/input | Objective and prediction |
|---|---|---|
| `protonet` | 100 ms statistics + mask; four 64-channel CNN blocks | Episodic CE; normalized class prototypes |
| `deepmetric` | 100 ms input; 128/256/512 double-convolution blocks; 1024-dimensional embedding | Batch-hard triplet, margin 0.2; maximum support cosine |
| `coda` | ProtoNet-style encoder | Episodic CE, contrastive loss, prototype consistency, and training cross-mode alignment |
| `cl_metaflow` | Four views: byte volumes, packet counts, mean length, joint statistics | 10,000 contrastive pretraining episodes, then 20,000 first-order meta episodes; five support-only SGD steps at lr 0.01 |
| `transformer` | 500 ms input; width 128, four heads, three layers, FFN 512 | Episodic CE; class prototypes |
| `global_amp` | Same AMP backbone, global embedding | Episodic CE, prefix augmentation, contrastive loss; SupportMax |
| `diagonal` / `soft_dtw` | AMP local/global fingerprints | Alternative correspondence operator, without PMA alignment supervision |

CL-MetaFlow episode evaluation adapts using support labels only; it does not use query labels for adaptation. Dedicated ablation configurations change fusion, partial views, loss terms, or aggregation and are trained independently.

## Running the experiment matrix

The shipped `experiments/matrix.json` contains **168 unique full-budget configurations**. Identical training configurations used by several analyses share one run. Regenerate the manifest and YAML files from the implementation defaults when needed:

```powershell
pea-vr matrix --output experiments
```

### Select Soft-DTW smoothing on validation data

```powershell
pea-vr run-matrix `
  --matrix experiments/matrix.json `
  --raw-root D:\PEA-VR-data\raw `
  --work-root D:\PEA-VR-work `
  --analyses soft_dtw_validation `
  --device cuda

pea-vr select-soft-dtw `
  --matrix experiments/matrix.json `
  --work-root D:\PEA-VR-work `
  --output experiments/selected_matrix.json
```

For each dataset, selection compares the three-seed mean best validation accuracy at gamma **0.01, 0.05, 0.1, 0.2, and 0.5**. All matched seeds and grid entries must be complete. The selected matrix assigns formal Soft-DTW comparisons to the selected gamma and records the chosen values. Test scores are not the selection criterion.

### Execute the selected matrix

```powershell
pea-vr run-matrix `
  --matrix experiments/selected_matrix.json `
  --raw-root D:\PEA-VR-data\raw `
  --work-root D:\PEA-VR-work `
  --device cuda
```

For each selected run, the runner constructs or loads the required prepared cohort, constructs missing validation/test protocols, trains or resumes from `latest.pt`, loads `best.pt`, and executes requested analyses. Existing completed analyses are tracked in `evaluation/complete.json`. Raw data must already be downloaded under `raw-root/longenough` and `raw-root/ydms`; matrix execution does not perform acquisition.

Held-out experiments receive distinct prepared/protocol directories such as `longenough_heldout_bw1`. Their normalizer and mode model are fitted without the excluded bandwidth. Their validation and test episodes select supports and queries from that held-out condition on disjoint validation/test identities.

The selected matrix stores resolved configuration paths. If relocating the checkout, regenerate it with `select-soft-dtw` against the relocated base matrix and the same completed validation runs.

| Analysis tag | Executed comparison |
|---|---|
| `table2` | Primary recognition with adapted baselines and alignment alternatives |
| `table3` | Known-coordinate correspondence diagnostics and supervision ablations |
| `table4` | Fusion, global-only, prefix, warp, alignment-loss, and contrastive-loss ablations |
| `table5` | Six separately trained support aggregation rules |
| `table6_heldout` | Independent bw1/bw8 held-out training and evaluation |
| `table6_open` | Five fixed known/unknown partitions per seed |
| `table7` | Actual parameter counts, storage, encoding, and ten-support matching measurements |
| `truncation` | Query fractions 0.2, 0.3, 0.5, 0.7, and 1.0 |
| `figure4` | Sixteen-cell bandwidth diagnostics |
| `figure9` | Token count, gap penalty, and global/local mixing sensitivity, using two seeds |
| `real_overlap` | Real-window player-reference overlap diagnostic |
| `soft_dtw_validation` | Validation-only smoothing grid |

Use `--analyses table2 table4`, `--names RUN_NAME`, or `--max-runs N` to select a scope. Selection limits the runs/analyses executed, not an individual run's 30,000-episode budget. The default runner uses one run at a time. Full matrix execution can be lengthy.

## Evaluation and diagnostics

### Closed-set recognition

```powershell
pea-vr evaluate `
  --checkpoint D:\PEA-VR-work\runs\longenough_pea_vr_20260814\best.pt `
  --data D:\PEA-VR-work\prepared\longenough `
  --protocols D:\PEA-VR-work\protocols\longenough\test `
  --output D:\PEA-VR-work\evaluation\main `
  --device cuda
```

Without `--scenarios`, evaluation uses every recognition scenario in the protocol manifest. To restrict it, use names such as `--scenarios cross_mode same_mode`. The checkpoint and prepared dataset fingerprints must match, and protocol files are verified before evaluation.

For each scenario/fraction, evaluation writes `SCENARIO_rhoVALUE.json` and `SCENARIO_rhoVALUE_episodes.jsonl`. The summary contains mean episode accuracy, episode sample standard deviation, standard error, episode count, configuration, checkpoint step, and dataset fingerprint. The JSONL records each episode's predictions, targets, and maximum candidate scores. Candidate confidence is a matching score, not a calibrated probability.

### Query truncation

```powershell
pea-vr evaluate `
  --checkpoint D:\PEA-VR-work\runs\longenough_pea_vr_20260814\best.pt `
  --data D:\PEA-VR-work\prepared\longenough `
  --protocols D:\PEA-VR-work\protocols\longenough\test `
  --scenarios cross_mode `
  --fractions 0.2 0.3 0.5 0.7 1 `
  --output D:\PEA-VR-work\evaluation\truncation `
  --device cuda
```

For fraction rho, queries use packets from `[0, 60*rho)` and recompute all method inputs before frozen normalization. Supports retain the full 60-second observation, and episode identities/session assignments stay fixed across fractions. This raw-packet truncation differs from the feature-mask prefix augmentation used during training.

### Correspondence with known source coordinates

```powershell
pea-vr correspondence --checkpoint D:\PEA-VR-work\runs\longenough_pea_vr_20260814\best.pt --data D:\PEA-VR-work\prepared\longenough --output D:\PEA-VR-work\evaluation\correspondence.json --device cuda
```

Each test session supplies two independently selected contiguous visible intervals, each covering a fraction in `[0.3, 1]`, with nonzero common support. Independent monotone warps give known source coordinates. The diagnostic reports alignment MAE in seconds, the fraction aligned within 2 seconds, and PMA overlap MAE. Per-session visibility intervals, token errors, and overlap measurements are saved in `correspondence.pairs.jsonl` for recomputation. Overlap MAE is reported only for PMA.

### Real-session overlap

```powershell
pea-vr real-overlap --checkpoint D:\PEA-VR-work\runs\longenough_pea_vr_20260814\best.pt --data D:\PEA-VR-work\prepared\longenough --start 30 --bootstrap 2000 --output D:\PEA-VR-work\evaluation\real_overlap.json --device cuda
```

This diagnostic requires the full LongEnough test cohort. It selects three valid sessions for each of 30 test identities and four bandwidth conditions: **360 real 60-second windows**, with **540 same-identity cross-bandwidth pairs**. The default window is `[30, 90]` seconds of the capture.

Player playback state and rate are integrated into playback progress, then buffer duration is added to estimate the delivered-media frontier. The reference overlap comes from common media intervals. Windows with insufficient traffic duration, player gaps, unannotated seeks, invalid frontiers, or zero media span are rejected. Insufficient valid sessions cause explicit failure with rejection records. Reported measurements include Spearman correlation, MAE, and identity-group bootstrap confidence intervals.

### Open-set recognition

```powershell
pea-vr open-set --checkpoint D:\PEA-VR-work\runs\longenough_pea_vr_20260814\best.pt --data D:\PEA-VR-work\prepared\longenough --protocols D:\PEA-VR-work\protocols\longenough\test --output D:\PEA-VR-work\evaluation\open_set.json --device cuda
```

The maximum candidate score is the known/unknown detection score. Evaluation records AUROC and FPR at the first ROC operating point with TPR at least 0.95, together with raw confidence and known/unknown labels for each partition. Three training seeds times five fixed partitions yield 15 measurements for aggregation. Operational rejection thresholds are calibrated separately on validation identities, as described below.

### Actual efficiency and storage

```powershell
pea-vr benchmark --checkpoint D:\PEA-VR-work\runs\longenough_pea_vr_20260814\best.pt --data D:\PEA-VR-work\prepared\longenough --warmup 100 --queries 1000 --output D:\PEA-VR-work\evaluation\benchmark.json --device cuda
```

Benchmarking uses FP32, query batch size 1, and ten supports from five identities. Encoding and matching are timed separately after warmup, with CUDA synchronization around timing. The report names the actual device and includes actual parameter counts, raw fingerprint tensor bytes, and serialized storage overhead. For the full method, storage consists of `z/U/r`; local matching projections can be reconstructed from `U` and the checkpoint.

### Inspect a fixed episode

```powershell
pea-vr case-study --checkpoint D:\PEA-VR-work\runs\longenough_pea_vr_20260814\best.pt --data D:\PEA-VR-work\prepared\longenough --protocols D:\PEA-VR-work\protocols\longenough\test --scenario mode_balanced --episode-index 0 --query-index 0 --output D:\PEA-VR-work\evaluation\case_study --device cuda
```

This exports local similarity, correspondence mass, candidate/support responsibilities, and traffic views as inspectable numeric artifacts and PDF/PNG figures. Use a scenario present in the supplied manifest; YDMS can use `random_mixed`.

## Reporting and independent metric checks

Matrix execution places results under each run's `evaluation/` directory. Aggregate, export, and plot them with:

```powershell
pea-vr collect --root D:\PEA-VR-work\runs --output D:\PEA-VR-work\summary
pea-vr tables --matrix experiments/selected_matrix.json --root D:\PEA-VR-work\runs --output D:\PEA-VR-work\tables
pea-vr plot --summary D:\PEA-VR-work\summary --output D:\PEA-VR-work\figures
```

`collect` groups compatible settings across seeds and saves recognition, open-set, and diagnostic summaries. Recognition tables use the mean and **sample standard deviation across training seeds**; episode standard error remains a separate within-run statistic. Open-set aggregation uses the 15 seed/partition measurements. Sensitivity figures use the two designated matched seeds.

`tables` reads results from the matrix run directories and checks training completion, the configured episode budget, compatible report configurations, expected evaluation episode counts, and required seed/partition counts. It writes `tables.json`, per-table CSV files, `missing.json`, and `status.json`. Missing or incomplete results make the default export fail. Use `--allow-incomplete` only to inspect progress; exported rows retain completeness flags.

The plotting command exports available measured sensitivity curves, primary truncation curves, bandwidth matrices, and real-overlap scatter/error distributions as PDF/PNG. It does not fill unavailable results with manuscript values. Standalone evaluation outputs can also be passed to `collect` as its root; strict matrix tables expect the matrix run layout.

Independently recompute recorded metrics:

```powershell
python scripts/recompute_metrics.py --root D:\PEA-VR-work\runs --output D:\PEA-VR-work\metric_audit.json
```

The script recomputes recognition accuracy from prediction/target records, open-set AUROC and FPR95 from raw confidence labels, and correspondence errors from saved pair records. It returns failure if no checkable measurements are found or if a recomputed measurement differs. Keep JSON summaries and their accompanying JSONL records together.

## New video enrollment and recognition

Enrollment uses a frozen checkpoint and labeled packet traces. It does not retrain the encoder or require player logs.

Create a UTF-8 CSV with columns `label,path,format`. Multiple rows with the same label provide multiple supports:

```csv
label,path,format
video_A,D:\PEA-VR-work\enrollment\video_A_session1.npz,npz
video_A,D:\PEA-VR-work\enrollment\video_A_session2.npz,npz
video_B,D:\PEA-VR-work\enrollment\video_B_session1.npz,npz
```

Supported formats are `npz`, `longenough`, and `ydms`. Relative paths are resolved against the CSV directory. An NPZ must contain aligned one-dimensional arrays:

| Array | Contract |
|---|---|
| `times` | Finite packet times in seconds; the earliest packet is shifted to time 0 |
| `directions` | 0 for downlink, 1 for uplink |
| `lengths` | Finite nonnegative packet lengths, using the checkpoint dataset's length convention |

Use the same observation and packet-length convention as training. The encoder applies the checkpoint's frozen normalizer.

```powershell
pea-vr enroll --checkpoint D:\PEA-VR-work\runs\longenough_pea_vr_20260814\best.pt --manifest D:\PEA-VR-work\enrollment.csv --output D:\PEA-VR-work\gallery.pt --device cuda

pea-vr calibrate --checkpoint D:\PEA-VR-work\runs\longenough_pea_vr_20260814\best.pt --data D:\PEA-VR-work\prepared\longenough --protocols D:\PEA-VR-work\protocols\longenough\validation --output D:\PEA-VR-work\threshold.json --device cuda

pea-vr recognize --checkpoint D:\PEA-VR-work\runs\longenough_pea_vr_20260814\best.pt --gallery D:\PEA-VR-work\gallery.pt --query D:\PEA-VR-work\query.npz --format npz --threshold D:\PEA-VR-work\threshold.json --output D:\PEA-VR-work\prediction.json --device cuda
```

Calibration uses the lower empirical 5th percentile of validation known-query confidence, targeting a known-query TPR of 0.95. The realized validation TPR is recorded. For YDMS, use its dataset/protocol paths; calibration defaults to `random_mixed`, while primary LongEnough defaults to `cross_mode`.

Omit `--threshold` for closed-set recognition. `--fraction` limits the query observation to that fraction of 60 seconds. The output contains the selected label, rejection flag, candidate scores, confidence, and optional threshold. A rejected query has a null label. Galleries and thresholds are bound to the checkpoint SHA-256; changing the checkpoint requires re-enrollment and recalibration.

## Automated tests

### Run the complete source test suite

```powershell
python -m pytest -q
```

The suite does not download public datasets or launch full-budget experiments. Temporary fixtures exercise boundary conditions and integration; public-data construction and scientific measurements are separate execution stages.

| Test module | Main checks |
|---|---|
| `test_alignment.py` | PMA reference equality, correspondence mass, gradient/second-gradient checks, Soft-DTW, CPU/CUDA behavior |
| `test_features_and_model.py` | Packet/bin boundaries, masks, normalization, fingerprints, partial views, aggregation, active gradients |
| `test_training_and_protocols.py` | Episode sampling, objective variants, chunk equivalence, CPU checkpoint resume |
| `test_data_and_evaluation.py` | Official formats, startup handling, archive safety, deterministic pooling, metrics, enrollment/evaluation integration |
| `test_complete_protocols.py` | Full protocol counts, mode/bandwidth partitions, open-set composition |
| `test_transfers_and_reporting.py` | Download interruption/resume, HTTP Range cache, independent ROC metrics, incomplete table rejection, plot grouping |

CUDA-dependent tests skip when CUDA is unavailable. To inspect skips or produce a machine-readable report:

```powershell
python -m pytest -q -rs --junitxml D:\PEA-VR-work\tests.xml
```

To run one focused check group:

```powershell
python -m pytest -q tests/test_alignment.py
python -m pytest -q tests/test_training_and_protocols.py
```

The mathematical tests compare PMA with an independent cell-wise recurrence and automatic derivatives, including second derivatives. Resume tests compare continuous and interrupted execution under the same numerical runtime. The data tests check training-index normalization and the distinction between unavailable bins and observed zero traffic.

### Test an installed package without the source-path override

The project pytest configuration normally adds `src` to Python's path. To verify a wheel installation, use a separate environment with the corresponding dependency set, install the wheel, and clear that override when running the unpacked tests:

```powershell
python -m pip wheel . --no-deps --wheel-dir D:\PEA-VR-work\wheels
```

In the separate test environment:

```powershell
python -m pip install --no-deps D:\PEA-VR-work\wheels\pea_vr-2.0.0-py3-none-any.whl
python -c "import pea_vr; print(pea_vr.__file__)"
python -m pytest -q -o pythonpath=
python -m pip check
```

The printed module path should point to that environment's `site-packages`. Run the test command from the unpacked project root; use a clean `PYTHONPATH`. This checks installed code rather than accidentally importing the working source tree.

## Troubleshooting and reproducibility

| Symptom | Action |
|---|---|
| `pea-vr` is not recognized | Activate the intended environment, invoke its full executable path, or use `python -m pea_vr` |
| CUDA is unavailable | Check the installed PyTorch build and device output; select the matching runtime or run with `--device cpu` |
| Strict cohort count mismatch | Inspect `cohort_inventory.json` and `excluded.jsonl`; check archive completeness, paired player files, offsets, and parser errors |
| Too few eligible identities for a scenario | Inspect retained modes/session counts; complete the required cohort rather than lowering protocol ways/shots |
| Prepared dataset already exists | Reuse it for training/evaluation, or choose a new preparation directory for a changed dataset |
| Checksum or fingerprint mismatch | Restore the matching immutable files or rebuild into a new directory; regenerate protocols for the rebuilt dataset |
| Resume configuration/runtime mismatch | Use the original configuration, dataset, implementation, validation files, and numerical runtime |
| CUDA memory exhaustion | Use smaller `query_chunk`, `alignment_chunk`, or `pair_chunk` in a new run configuration; retain episode ways/shots for the intended comparison |
| Missing `best.pt` | Confirm validation protocols were supplied and the run reached a validation boundary |
| Table export is incomplete | Read `missing.json` and `status.json`; complete the required training seeds and evaluation reports |
| Real-overlap diagnostic rejects sessions | Inspect the rejection file and required player observations for the 30–90 second window |
| Gallery/threshold checkpoint mismatch | Re-enroll or recalibrate with the checkpoint used for recognition |

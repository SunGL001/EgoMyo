# EgoMyo | MyoBridge

[![Website](https://img.shields.io/badge/Project-Website-2ea44f)](https://egomyo.github.io/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-EgoMyo-yellow)](https://huggingface.co/datasets/long2333/EgoMyo)
[![Code](https://img.shields.io/badge/Code-SunGL001%2FEgoMyo-181717?logo=github)](https://github.com/SunGL001/EgoMyo)
[![Paper](https://img.shields.io/badge/arXiv-coming%20soon-lightgrey)](#paper)

Reference implementation of **MyoBridge**, a multimodal model that predicts
hand pose, contact and regional press force from egocentric vision and a causal
18-channel sEMG window. This repository contains the training-ready model code,
data-layout checks and evaluation scripts. The EgoMyo dataset is hosted
separately; licensed MANO and WiLoR weights are not redistributed here.

| Resource | Link |
| --- | --- |
| Project website and qualitative results | [egomyo.github.io](https://egomyo.github.io/) |
| Dataset repository | [🤗 long2333/EgoMyo](https://huggingface.co/datasets/long2333/EgoMyo) |
| Model code | [SunGL001/EgoMyo](https://github.com/SunGL001/EgoMyo) |
| Paper | [arXiv link pending](#paper) |

> **Dataset availability:** follow the Hugging Face dataset card for current
> access, release, and license terms. Do not redistribute recordings or derived
> annotations unless you have the required participant consent and release
> authorization.

## Layout

```
myobridge/
├── myobridge/
│   ├── mano.py          MANO buffers and native linear blend skinning
│   ├── components.py    EMG encoder, mixture-of-transformer block, routing, force field
│   ├── model.py         final MyoBridge architecture and model builder
│   ├── losses.py        multi-task objective and EMG augmentation
│   ├── data.py          trial dataset, block sampler, collate function
│   ├── splits.py        cross-trial, cross-session and cross-subject splits
│   ├── metrics.py       geometry, contact and force metrics
│   └── utils.py         seeding, checkpointing, partition loading
├── scripts/
│   ├── prepare_mano.py            convert an official MANO file to buffers
│   ├── make_splits.py             freeze the three evaluation splits
│   ├── compute_normalization.py   TRAIN-only EMG statistics
│   ├── extract_wilor_tokens.py    cache frozen WiLoR tokens
│   ├── extract_dino_features.py   optional cached DINOv2 features
│   ├── train.py                   training driver
│   ├── evaluate.py                held-out evaluation
│   └── examples/                  example command lines
├── third_party/wilor_adapter.py   interface to an official WiLoR checkout
└── tests/                         model, split and metric tests
```

## Environment

```bash
conda env create -f environment.yml
conda activate myobridge
```

or

```bash
pip install -r requirements.txt
```

## Dataset

The dataset repository and this code repository are separate. Download the
dataset from the Hugging Face Hub; the helper supports authenticated/gated
repositories through the Hugging Face CLI login and optional file filters.
This downloads data only when you run it; cloning this code repository never
downloads the dataset.

```bash
pip install -r requirements.txt
hf auth login
python scripts/download_dataset.py --local-dir data/egomyo
python scripts/validate_dataset.py --data data/egomyo --visual wilor
```

To fetch only selected files from the Hub, repeat `--allow-pattern`:

```bash
python scripts/download_dataset.py --local-dir data/egomyo \
  --allow-pattern README.md --allow-pattern membership.json
```

The training code consumes a **model-ready export**, not the acquisition/source
package. Its expected layout is:

```text
data/egomyo/
├── membership.json
├── normalization.json
├── splits.json
├── features/
│   ├── wilor/<trial_id>.npy
│   └── dino/<trial_id>.npy          # optional DINOv2 variant
└── targets/<trial_id>.npz
```

`membership.json` records anonymized trial IDs and split metadata;
`normalization.json` contains TRAIN-only mean/std for 18 EMG channels;
`splits.json` freezes the three protocols. Each target archive contains aligned
EMG windows, pose/mesh, regional and vertex contact/force targets, validity
masks and handedness. The loader expects 400-sample EMG windows, 21 joints,
778 mesh vertices and seven anatomical regions. The default WiLoR feature is
`[frames, 13, 1280]`; optional DINOv2 features are `[frames, 34, 768]`.

Run the validator before training; it checks metadata, protocol leakage,
required per-trial files and feature shapes. A source/acquisition archive must
first be transformed and reviewed against this documented export schema; do
not point training scripts at the raw acquisition package. The release
checklist and an unapproved dataset-card draft are in `docs/`.

Once validated, train as shown below. For reproducibility, record the dataset
revision/commit shown on the Hugging Face dataset page with each experiment.

Training uses BF16 autocast and MATH SDPA on CUDA. The code also runs on CPU in
FP32, which is useful for the unit tests but not for full training.

The MANO model is licensed and must be obtained from its license holder. Convert
your official right-hand model file once:

```bash
python scripts/prepare_mano.py --mano-model /path/to/MANO_RIGHT.pkl --out assets/mano_buffers.npz
```

The buffer archive is validated on load. The anatomical partition over the 778
MANO vertices is part of the dataset release and is loaded from
`anatomical_partition.npz` with a single int64 array `partition` holding region
ids `-1` (unassigned) or `0..6`.

## Dataset format

The dataset root is expected to contain:

```
<data>/
├── membership.json          trial ids and their capture metadata
├── normalization.json       per-channel EMG mean and standard deviation
├── features/
│   ├── wilor/<trial_id>.npy float16 [frames, 13, 1280]
│   └── dino/<trial_id>.npy  float16 [frames, 34, 768]   (optional)
├── targets/<trial_id>.npz   frames of supervision
└── crops/<trial_id>/wilor.npy   RGB crops [frames, 3, 256, 256]  (extraction only)
```

`membership.json` holds one entry per trial:

```json
{"trials": [{"trial_id": "...", "package": "p01-R", "participant": "p01",
             "session": 1, "action": 7, "repetition": 2, "side": "R",
             "frames": 158}]}
```

`targets.npz` holds, per frame, `emg` `[18, 400]` raw ADC windows ending at the
observed frame, root-relative `mesh` `[778, 3]` and `joints` `[21, 3]` in metres,
`vertex_force` `[778]` and `vertex_contact` `[778]`, `regional_force` `[7]` and
`regional_contact` `[7]`, `surface_regional_force` `[7]`, and the validity masks
`mesh_valid`, `joint_valid`, `force_valid`, `contact_valid`, `surface_valid`,
`distribution_valid`, plus `input_valid` and `is_left`.

Feature extraction is optional if the release already ships cached features.
WiLoR is third party software. Install it from its official repository and point
`--adapter` at a module implementing the interface in
`third_party/wilor_adapter.py`:

```bash
python scripts/extract_wilor_tokens.py --data <data> --crops <data>/crops \
  --adapter third_party/wilor_adapter.py --checkpoint /path/to/wilor.ckpt
```

The optional DINOv2 variant uses `scripts/extract_dino_features.py` and is not
used by the default configuration.

## Splits

```bash
python scripts/make_splits.py --membership <data>/membership.json \
  --split-seed 0 --out <data>/splits.json
```

The three protocols separate increasing amounts of capture variation.

| Protocol | Train | Test |
|---|---|---|
| `cross_trial` | three repetitions of every capture stratum | the other two repetitions |
| `cross_session` | session 1 | session 2 |
| `cross_subject` | 70% of participants, both hands and sessions | the remaining participants |

Every ranking uses a salted SHA-256 key over the trial or participant id, so the
split is reproducible from `membership.json` alone. The splitter rejects
incomplete repetition strata, overlapping partitions and participant leakage.

Then compute TRAIN-only EMG statistics:

```bash
python scripts/compute_normalization.py --data <data> --splits <data>/splits.json \
  --protocol cross_trial --out <data>/normalization.json
```

## Training

Default configuration, frozen WiLoR tokens with no pose or shape prior:

```bash
DATA=<data> MANO=assets/mano_buffers.npz PARTITION=<data>/anatomical_partition.npz \
  bash scripts/examples/train_wilor.sh
```

or directly

```bash
python scripts/train.py --data <data> --splits <data>/splits.json \
  --protocol cross_trial --mano assets/mano_buffers.npz \
  --partition <data>/anatomical_partition.npz --visual wilor \
  --seed 0 --epochs 20 --batch 256 --block 128 --lr 5e-4 --out runs/cross_trial_wilor_s0
```

Optimization uses AdamW with weight decay 0.01, gradient clipping at 5, a
two-epoch warmup followed by a cosine schedule, and a batch of 256 frames built
from shuffled 128-frame trial blocks. The DINOv2 variant trains only the
residual adapter at 5e-5 while the downstream network keeps 5e-4.

Checkpoints store learned weights only. The licensed MANO buffers are read from
the local model file at every run and are never written into a checkpoint.

## Evaluation

```bash
python scripts/evaluate.py --data <data> --splits <data>/splits.json \
  --protocol cross_trial --mano assets/mano_buffers.npz \
  --partition <data>/anatomical_partition.npz \
  --checkpoint runs/cross_trial_wilor_s0/checkpoint.pt --visual wilor --seed 0 \
  --out runs/cross_trial_wilor_s0/test_metrics.json
```

Weights come from the final training epoch, without any selection on test
performance. Reported metrics are root-relative MPJPE and PVE in millimetres,
region contact macro F1 at a 0.5 threshold, surface contact F1 with a fixed
5 mm edge-geodesic neighborhood on the rest mesh, and regional and per-vertex
force MAE in estimated newtons. Force values are model estimates rather than
calibrated sensor readings.

To aggregate several seeds, average within trial first and then across trials,
and report the sample standard deviation across seeds.

## Tests

```bash
python -m pytest tests
```

The tests build synthetic MANO buffers of the correct layout, so they exercise
the full forward path and the objective without the licensed model.

## Scope

Included: environment, split construction, cached-feature loading, the final
model, training and evaluation. Not included: baseline implementations,
analysis scripts, dataset files, pretrained WiLoR weights and the MANO model.

## Paper

The arXiv preprint link will be added here when available. Until then, use the
[project website](https://egomyo.github.io/) for project materials and updates.

# EgoMyo | MyoBridge

[![Paper](https://img.shields.io/badge/arXiv-2611.34224-b31b1b)](https://arxiv.org/abs/2611.34224)
[![Website](https://img.shields.io/badge/Project-Website-2ea44f)](https://egomyo.github.io/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-EgoMyo-yellow)](https://huggingface.co/datasets/long2333/EgoMyo)
[![Code](https://img.shields.io/badge/Code-SunGL001%2FEgoMyo-181717?logo=github)](https://github.com/SunGL001/EgoMyo)

Official code for **[EgoMyo: Physical Hand State Estimation from Egocentric
Video and Wrist Electromyography](https://arxiv.org/abs/2611.34224)**.

**EgoMyo** provides egocentric video, wrist surface electromyography (sEMG),
and hand pose, contact and force annotations for studying physical hand
interaction. The public dataset contains **100 hand packages and 33,989 trials**.

**MyoBridge** estimates hand pose, contact and force jointly from video and a
causal 18-channel sEMG window. It fuses the modalities at the hand-region level,
uses regional gates to control the contribution of sEMG, and distributes
regional forces over predicted contact vertices. The default implementation
uses **frozen WiLoR visual features without explicit WiLoR pose/shape priors**.

## What Is Included

- **MyoBridge:** model, training losses, fixed-epoch training and evaluation.
- **Data preparation:** dataset download, raw-to-target conversion, video-based
  visual feature extraction and TRAIN-only EMG normalization.
- **Baselines:** task-adapted comparison models and a shared training/evaluation
  interface in [`baselines/`](baselines/README.md).
- **Documentation:** [input/target definitions](docs/data-field-mapping.md),
  [end-to-end preparation](docs/raw-to-targets.md), and tests.

Recordings are hosted on [Hugging Face](https://huggingface.co/datasets/long2333/EgoMyo).
External pretrained weights and licensed MANO assets must be obtained separately.

## Installation

```bash
git clone https://github.com/SunGL001/EgoMyo.git
cd EgoMyo
conda env create -f environment.yml
conda activate myobridge
```

Alternatively, install the dependencies into an existing compatible environment:

```bash
pip install -r requirements.txt
```

WiLoR feature extraction additionally requires an official
[WiLoR checkout](https://github.com/rolpotamkin/WiLoR), its dependencies,
checkpoint, configuration and hand detector. Obtain the right-hand MANO model
under its license and convert it once:

```bash
python scripts/prepare_mano.py --mano-model /path/to/MANO_RIGHT.pkl \
  --out assets/mano_buffers.npz
```

## Data and Model Inputs

| Role | Dataset source | Prepared representation |
| --- | --- | --- |
| Visual input | `ego/ego.mp4` and video timestamps | Frozen WiLoR tokens `[frames, 13, 1280]`; optional DINOv2 features `[frames, 34, 768]` |
| sEMG input | `emg/emg_raw.h5`, raw 18-channel ADC and timestamps | Causal `[18, 400]` windows per frame, normalized using TRAIN only |
| Pose/mesh supervision | Original `contact/mano.npz` | Wrist-relative 21 joints and 778 vertices in metres, with validity masks |
| Regional contact/force supervision | Corrected/completed **pressure2** in `force/pressure_completed.h5` | Seven-region force estimates and contact labels derived using a `> 0.2` estimated-N threshold |
| Surface contact/force supervision | `contact/contact_surface_completed.h5` | Mapped vertex contact/force targets and mapping-validity masks |

**Pose, contact and force labels are supervision, not model inputs.** The default
model does not receive ground-truth geometry or WiLoR pose/shape predictions.
Optional completed MANO meshes are for preview, not replacements for original
pose supervision. Pressure2 retains completion and quality flags; force values
are estimates, not independently calibrated measurements. See
[field definitions and masks](docs/data-field-mapping.md).

## Prepare the Dataset

### 1. Download and extract

```bash
python scripts/download_dataset.py --local-dir data/egomyo-release
mkdir -p data/egomyo-raw
for archive in data/egomyo-release/data/raw/*.tar; do
  tar -xf "$archive" -C data/egomyo-raw
done
```

For a smaller download, repeat `--allow-pattern` to select archives and metadata:

```bash
python scripts/download_dataset.py --local-dir data/egomyo-release \
  --allow-pattern README.md --allow-pattern 'metadata/*' \
  --allow-pattern 'data/raw/0929-1-L.tar' --allow-pattern release.json
```

Available archives and SHA-256 checksums are listed in the dataset's
[`release.json`](https://huggingface.co/datasets/long2333/EgoMyo/blob/main/release.json)
and `metadata/packages/`. Verify the checksums before extraction and record the
dataset revision used. All archives extract into the same raw root. A single
package is not sufficient to train on the full published split. Only corrected
**pressure2** is distributed; no older pressure files are needed.

### 2. Convert recordings and annotations to aligned targets

```bash
python scripts/prepare_targets.py --source-root data/egomyo-raw \
  --membership data/egomyo-release/metadata/membership.json \
  --out data/egomyo-prepared --include-long-emg
cp data/egomyo-release/metadata/splits.json data/egomyo-prepared/splits.json
```

The output directory must not already exist. The converter creates aligned
targets, a video frame map, anatomical region assignments, validity masks and
preparation receipts. `--include-long-emg` also exports 2000-sample windows for
long-history baselines; MyoBridge uses the 400-sample windows.

### 3. Extract features and compute normalization

```bash
DATA=data/egomyo-prepared
python scripts/extract_wilor_from_video.py --data "$DATA" \
  --frame-map "$DATA/frame-map.json" --wilor-root /path/to/WiLoR \
  --checkpoint /path/to/wilor_final.ckpt --config /path/to/model_config.yaml \
  --detector /path/to/detector.pt
python scripts/compute_normalization.py --data "$DATA" --splits "$DATA/splits.json" \
  --protocol cross_trial --out "$DATA/normalization-cross_trial.json"
python scripts/validate_dataset.py --data "$DATA" \
  --normalization "$DATA/normalization-cross_trial.json"
```

Feature extraction uses the detector and video, not ground-truth pose, to
construct hand crops. The prepared directory contains:

```text
egomyo-prepared/
├── membership.json
├── splits.json
├── anatomical_partition.npz
├── normalization-cross_trial.json
├── frame-map.json
├── features/wilor/<trial_id>.npy
└── targets/<trial_id>.npz
```

Detailed alignment rules, missing-value handling, pressure2 semantics and
optional DINOv2 extraction are documented in
[`docs/raw-to-targets.md`](docs/raw-to-targets.md) and
[`docs/data-field-mapping.md`](docs/data-field-mapping.md).

## Train and Evaluate

Train the WiLoR-feature model for 20 epochs:

```bash
DATA=data/egomyo-prepared
python scripts/train.py --data "$DATA" --splits "$DATA/splits.json" \
  --protocol cross_trial --normalization "$DATA/normalization-cross_trial.json" \
  --mano assets/mano_buffers.npz --partition "$DATA/anatomical_partition.npz" \
  --visual wilor --seed 0 --epochs 20 --batch 256 --block 128 --lr 5e-4 \
  --out runs/cross_trial_wilor_s0
```

Evaluate the final checkpoint on TEST, using the same normalization:

```bash
python scripts/evaluate.py --data "$DATA" --splits "$DATA/splits.json" \
  --protocol cross_trial --normalization "$DATA/normalization-cross_trial.json" \
  --mano assets/mano_buffers.npz --partition "$DATA/anatomical_partition.npz" \
  --checkpoint runs/cross_trial_wilor_s0/checkpoint.pt --visual wilor --seed 0 \
  --out runs/cross_trial_wilor_s0/test_metrics.json
```

Repeat with seeds `0`, `1`, `2`, keeping the data split fixed. Adjust batch size
and workers to your hardware. CUDA training uses BF16; CPU execution uses FP32.
Checkpoints contain learned weights, not licensed MANO buffers.

The public frozen **cross-trial** split contains **13,599 TRAIN / 6,800 validation /
13,590 TEST trials**. `metadata/splits_final.json` merges TRAIN and validation
for final refit. Select the configuration on validation before refitting; do
not use TEST for tuning or normalization. Code also supports cross-session and
cross-subject protocols, but these require suitable reviewed session/participant
metadata and their own frozen splits; the supplied public split is cross-trial.
Compute normalization separately for each protocol or refit training partition.

Metrics include MPJPE/PVE in millimetres, regional and surface contact F1, and
regional and vertex force MAE in estimated newtons. See
[`docs/data-field-mapping.md`](docs/data-field-mapping.md) for masks and metric
definitions. Report means and sample standard deviations across seeds.

## Baselines and Tests

[`baselines/README.md`](baselines/README.md) provides input requirements, external
dependencies and commands for OpenTouch, EgoTouch, EgoTactile, EgoPressure, MoT,
HBridge, NeuroPose, EMGFormer, vemg2pose, HaMeR and WiLoR. It identifies which
implementations are task-adapted comparisons and which use native external
components. Methods share the prepared targets, split files and metrics.

```bash
python -m baselines.run --help
python -m pytest tests
```

## Citation

```bibtex
@article{sun2026egomyo,
  title={EgoMyo: Physical Hand State Estimation from Egocentric Video and Wrist Electromyography},
  author={Sun, Guanglong and Zhou, Kanglei and Wang, Ziteng and Wang, Nana and Tan, Hengkai and Liu, Xiaolong and Su, Hang and Zhong, Yi and Zhu, Jun},
  journal={arXiv preprint arXiv:2611.34224},
  year={2026},
  url={https://arxiv.org/abs/2611.34224}
}
```

For dataset details and terms, see the
[dataset card](https://huggingface.co/datasets/long2333/EgoMyo).
Questions and bug reports are welcome in [Issues](https://github.com/SunGL001/EgoMyo/issues).

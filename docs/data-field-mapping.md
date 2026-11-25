# MyoBridge data-field and supervision map

This document describes what the released training/evaluation code actually
reads. The model-ready export is a derived input package; it is not the raw
acquisition directory. File names below are relative to the dataset root.

## Signal path

```text
RGB hand crop (feature extraction only)
  └─ frozen WiLoR backbone → 12 pooled spatial tokens + 1 global token
                                ┐
18-channel causal sEMG window ──┴─ multimodal MyoBridge → MANO pose/shape
                                                     → joints + mesh
                                                     → regional/surface contact
                                                     → regional/surface force
```

In the default `--visual wilor` run, the model receives the **cached WiLoR
feature tensor**, not decoded video frames, WiLoR pose estimates, or WiLoR shape
estimates. The WiLoR backbone is frozen during extraction and is not part of the
training graph. `--visual dino` is an alternative visual input, not an
additional stream: it consumes cached DINOv2 tokens and trains the code's
residual feature adapter. The repository's DINO extractor freezes DINOv2.

The model predicts MANO rotation/shape parameters itself and passes those
predictions through the differentiable MANO layer to obtain its mesh and
joints. MANO is a geometric decoder here, not an input pose/shape prior.

## Model inputs and metadata

| Input | Source file/field | Shape per frame | What the code does |
| --- | --- | --- | --- |
| sEMG | `targets/<trial_id>.npz:emg` | `[18, 400]` | Uses the 18 electrode channels over a 400-sample causal window ending at the observed frame. Applies per-channel train-split z-score normalization, then training-only gain/offset noise and channel/window dropout. |
| WiLoR vision (default) | `features/wilor/<trial_id>.npy` | `[13, 1280]` | Uses 12 tokens from adaptive 4×3 spatial average pooling plus one global spatial-mean token. The feature projection is trained; the feature backbone is not. |
| DINOv2 vision (optional) | `features/dino/<trial_id>.npy` | `[34, 768]` | Uses a class token, patch-token mean and 32 tokens from adaptive 4×8 patch-grid pooling. Select with `--visual dino`; it replaces WiLoR. |
| Handedness | `targets/<trial_id>.npz:is_left` | scalar boolean | Selects left/right MANO handling. It is metadata supplied to the forward pass, not a prediction target. |

For WiLoR, raw MP4 frames can be converted directly to the released feature
shape by `scripts/extract_wilor_from_video.py`. It uses the official WiLoR
YOLO detector, official `ViTDetDataset` crop/resize/flip/normalization path,
and the frozen WiLoR backbone. It selects one detection of the declared
handedness per requested frame; zero or multiple matching detections stop the
run instead of silently inserting a missing crop. Its detector boxes are
independent predictions, never derived from GT joints, MANO parameters, or
contact labels. The resulting `[T,13,1280]` features correspond exactly to the
explicit frame indices in the supplied frame map.

The frame map is a reviewed synchronization artifact and must not be guessed
from trial ordering. Example:

```json
{
  "trials": {
    "trial_001": {
      "video": "raw/session_01/trial_001.mp4",
      "side": "R",
      "frame_indices": [12, 15, 18]
    }
  }
}
```

Each `frame_indices` list must have exactly `membership.json`'s `frames` count.
Indices are zero-based decoded video-frame indices, in target-row order; each
must be reviewed against the approved video/EMG/annotation synchronization
procedure. The script does not align timestamps or resample EMG. Run it from
the release repository, with the official WiLoR checkout, its detector and
WiLoR checkpoint/config available locally:

```bash
python scripts/extract_wilor_from_video.py \
  --data data/egomyo --frame-map /path/to/reviewed-frame-map.json \
  --wilor-root /path/to/WiLoR --checkpoint /path/to/wilor_final.ckpt \
  --config /path/to/model_config.yaml \
  --detector /path/to/detector.pt --device cuda
```

The script writes `features/wilor/<trial_id>.npy` and an extraction record
containing checkpoint hashes and preprocessing settings. The WiLoR code,
checkpoints, MANO assets and detector are external licensed dependencies and
are not bundled here. The older `scripts/extract_wilor_tokens.py` remains for
users who already have RGB crops `[T,3,256,256]` in `[0,1]`; its adapter applies
the official WiLoR normalization and horizontal 32-pixel trim. DINOv2
extraction uses ImageNet-normalized crops (`[T,3,224,224]`) via
`scripts/extract_dino_features.py`. Add `--export-baseline-crops` to the raw
WiLoR video extractor to generate those crops using the same detected hand,
without GT pose/pressure-based cropping.

The full annotation-aware conversion is provided by
[`scripts/prepare_targets.py`](../scripts/prepare_targets.py), documented in
[`raw-to-targets.md`](raw-to-targets.md). It defaults to **completed pressure2**,
exports aligned EMG/pose/contact/force targets and validity masks, and records
source checksums and the exact conversion policy. pressure2 regional contact
is derived from unsmoothed completed regional force >0.2 estimated N; unresolved
and missing-source rows remain invalid. Surface labels use completed per-face
force with explicit spatial-mapping validity. Original MANO geometry/validity is
preserved, not replaced by a completed preview mesh. Imputed force is derived
supervision, not a raw measurement. Protocol splits and TRAIN-only normalization
remain separate, explicit steps. See [`../baselines/README.md`](../baselines/README.md)
for the baseline inputs, including optional causal 2000-sample raw EMG windows.

`membership.json` supplies `trial_id` and `frames` to the loader. The split
builder additionally uses `participant`, `package`, `session`, `action` and
`repetition`; those fields define the split groups and are not model features.
The code does not read raw camera frames during train/evaluation.

### EMG normalization and validity

`scripts/compute_normalization.py` computes one mean and standard deviation per
electrode using only the selected protocol's training trials. It includes
windows selected by `input_valid` when that optional field exists; if the field
is absent, it uses every window. The resulting 18-element arrays are saved in
`normalization.json` and reused for validation/test and all seeds. Training
augmentation applies a log-normal channel gain (log standard deviation 0.20),
Gaussian channel offset (standard deviation 0.10), independent 10% electrode
dropout and 10% whole-window EMG dropout. These are training augmentations, not
changes to the stored signal.

**Important implementation detail:** `TrialDataset` does not load `input_valid`
or filter invalid-input frames from training/evaluation. It is currently used
only by the normalization script. Invalid target annotations are masked using
the target validity fields below. Reproducing this code version means retaining
that behavior; changing it changes the training sample set and requires new
results.

Compute a distinct normalization file for each evaluation protocol, then pass
that exact file to both `scripts/train.py --normalization` and
`scripts/evaluate.py --normalization`. The CLI checks the protocol metadata
written by `compute_normalization.py` and rejects a cross-protocol mismatch.

The code defines 18 channels and window length, but does not define the
electrode/channel names or order, physical units, sampling rate, hardware
filtering, or the conversion from video timestamps to EMG samples. Those facts
must be supplied in the dataset metadata; do not infer them from tensor shape.

## Supervision fields

All target arrays are frame-aligned in `targets/<trial_id>.npz`.

| Target | Expected shape | Used for |
| --- | --- | --- |
| `mesh` | `[T,778,3]`, metres | MANO surface-vertex geometry supervision and PVE evaluation. |
| `joints` | `[T,21,3]`, metres | Root-relative joint geometry supervision and MPJPE evaluation. The metric re-aligns both prediction and target to joint 0. |
| `vertex_contact` | `[T,778]`, binary | Per-vertex contact-logit binary cross entropy plus Dice loss; also surface-contact F1 evaluation. |
| `regional_contact` | `[T,7]`, binary | Per-region contact-probability binary cross entropy and regional contact F1. |
| `regional_force` | `[T,7]`, force units recorded by the release | Per-region absolute-error force loss and regional-force MAE. Current metric naming treats the values as estimated, uncalibrated newtons; the release must document their derivation/calibration. |
| `vertex_force` | `[T,778]`, force units recorded by the release | Defines the target within-region force distribution. The model learns its normalized allocation with cross entropy; there is no separate direct vertex-force magnitude loss. Per-vertex force MAE is an evaluation metric. |
| `surface_regional_force` | `[T,7]`, force units recorded by the release | Denominator for normalizing `vertex_force` into a within-region allocation target. It is not a separate model input or prediction head. |

The force-distribution loss is applied only to regions that are valid,
contact-positive, have positive `surface_regional_force`, and have a valid
distribution annotation. The predicted vertex force is constructed from the
predicted regional force and predicted within-region allocation.

## Validity masks and handedness

| Field | Shape | Meaning in the code |
| --- | --- | --- |
| `mesh_valid` | `[T]` | Enables mesh geometry supervision. |
| `joint_valid` | `[T,21]` | Enables supervision/evaluation for each joint. |
| `surface_valid` | `[T]` | Enables surface vertex contact and distribution support. |
| `contact_valid` | `[T,7]` | Enables regional contact supervision/evaluation. |
| `force_valid` | `[T,7]` | Enables regional force supervision/evaluation. |
| `distribution_valid` | `[T,7]` | Enables within-region force-allocation supervision when the region is also active. |
| `is_left` | `[T]` | Handedness metadata passed to the MANO decoder. |
| `input_valid` | `[T]`, optional | Used by EMG normalization only; see the behavior note above. |

Every mask's exact definition, including how missing/ambiguous annotations are
marked, belongs in the released dataset card. The code treats validity masks as
authoritative and does not impute missing pose/contact/force labels.

## Objective and metrics

`scripts/train.py` uses four learned-uncertainty-weighted objective groups:

1. **Geometry:** root-relative mesh and joint L1 errors in metres, scaled by
   100. Mesh and joint terms share the geometry uncertainty weight.
2. **Contact:** vertex contact-logit BCE plus Dice loss, and regional contact
   BCE, each using the corresponding validity mask.
3. **Regional force:** masked L1 error between predicted and target regional
   force.
4. **Force allocation:** cross entropy from the normalized target vertex-force
   distribution to the predicted within-region distribution, using only valid,
   active contact regions.

`scripts/evaluate.py` evaluates the frozen final-epoch checkpoint on the
selected held-out split: root-relative MPJPE/PVE in millimetres, regional
contact macro-F1 at 0.5, regional-force MAE, surface-contact F1 with the fixed
5 mm rest-mesh edge-geodesic neighborhood, and contact-vertex force MAE. It
does not select checkpoints using the test split. Force metrics are reported
as estimates, not calibrated sensor measurements.

## Reproduction facts the release must provide

The code fixes tensor dimensions and its transforms, but the following
dataset-specific facts are not encoded in this repository. A release is not
fully reproducible until the dataset card/metadata records them:

- Exact 18-electrode names/order, units, sample rate, filters and EMG-to-video
  timestamp/window alignment.
- Video frame rate/timebase, crop-generation procedure, RGB/color convention,
  resizing/normalization and frame-to-feature alignment.
- Names and ordering of all seven regions, anatomical partition version/hash,
  MANO model version/topology and the joint-index convention.
- Pose coordinate frame and units; contact-label threshold/definition; force
  calculation, units, calibration status and valid/missing-value semantics.
- Dataset release revision and file checksums; exact WiLoR/DINO checkpoint
  identifiers and hashes for regenerating features; MANO/partition file hashes.
- Python/PyTorch/CUDA and third-party extractor versions, code commit, split
  seed, normalization protocol and training command/hyperparameters.

The data-card draft deliberately leaves these acquisition/release facts as
TODOs rather than fabricating values. Preserve the dataset revision, code
commit, external asset hashes and environment with every reported run.

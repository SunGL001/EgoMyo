# EgoMyo baseline reproduction

[Project](https://egomyo.github.io/) ·
[Dataset](https://huggingface.co/datasets/long2333/EgoMyo) ·
[Code](https://github.com/SunGL001/EgoMyo)

This directory provides **task-adapted comparisons on EgoMyo**, a shared
training/evaluation entry point, and explicit external-asset interfaces.
It is not a claim to reproduce each method's original paper/dataset protocol.
No target pose, pressure or contact annotation enters a model forward.
All methods use the same prepared **pressure2** supervision and frozen splits.

## Implementations and inputs

| CLI method | Model implemented here | Inference inputs | External dependencies |
| --- | --- | --- | --- |
| `opentouch` | Owner-adapted spatial-token transformer (one layer) + contact/force MLP | Frozen DINO `[34,768]` | DINO features |
| `egotouch` | Owner-adapted spatial-token transformer (two layers), 384-wide readout | Frozen DINO `[34,768]` | DINO features |
| `egotactile` | Owner-adapted PressureFormer-style token transformer + MLP | Frozen DINO `[34,768]` | DINO features |
| `egopressure` | Explicit **approximation**, DINO class + mean tokens → two-layer MLP | Frozen DINO `[34,768]` | DINO features; not official EgoPressure source |
| `mot` | Six modality-expert attention blocks with joint attention at every layer | Frozen DINO + normalized EMG `[18,400]` | DINO features |
| `hbridge` | Unequal-width branches, middle-layer joint attention, 16 semantic reconstruction tokens | Frozen DINO + normalized EMG `[18,400]` | DINO features |
| `neuropose` | Pinned EgoEMG NeuroPose encoder/residual/decoder, signed final decoder, endpoint Cartesian readout | Past raw EMG `[18,2000]` | Pinned external EgoEMG components |
| `emgformer` | Pinned TDS + RoPE decoder + attention pooling, visual pose + small residual fusion | Past raw EMG `[18,2000]` + frozen DINO | Pinned external EgoEMG components |
| `emg2pose` | vemg2pose adaptation: original Meta TDS, zero-initialized autoregressive Cartesian state, learned mesh readout | Past raw EMG `[18,2000]` | Pinned external Meta network source |
| `hamer` | External HaMeR native RGB geometry + attached physical readouts | Detected/canonicalized RGB crops and crop metadata | Official HaMeR checkout, publisher checkpoint, licensed MANO |
| `wilor` | External WiLoR native RGB geometry + attached physical readouts | Detected/canonicalized RGB crops and crop metadata | Official WiLoR checkout, publisher checkpoint, licensed MANO |

`cached.py` ports the owner's adapted visual and MoT/HBridge task architectures;
these are not diffusion/image/language generators or official OpenTouch,
EgoTouch, EgoTactile or EgoPressure implementations. `emg.py` loads original
components externally. Native geometry/contact/force paths are retained;
direct linear readouts provide tasks not present in a native method. Geometry
outputs are root-relative metres in the delivered 21-joint/778-vertex order.
Left-hand crops/geometry use the canonical-right convention; the wrapper
reflects predicted x back to the delivered hand side before scoring.

## 1. Prepare data and visual caches

Follow [`../docs/raw-to-targets.md`](../docs/raw-to-targets.md). For EMG baselines
include `--include-long-emg` when preparing targets: 400 samples are never
zero-extended and presented as a real 2000-sample history. Long-window padding
and acquisition gaps invalidate the corresponding supervision. NeuroPose uses
explicit computational padding to 32 channels/2240 positions but reads position
1999 (the last real input sample), never the padded tail. EMGFormer adds four
right zeros to preserve endpoint coverage. Meta state initialization is zero,
not GT pose. These three models consume raw ADC, not normalized EMG.

For cached DINO baselines and RGB methods, export the detector-generated crops:

```bash
DATA=/data/EgoMyo/prepared-pressure2
python scripts/extract_wilor_from_video.py --data "$DATA" \
  --frame-map "$DATA/frame-map.json" --wilor-root /opt/WiLoR \
  --checkpoint /assets/wilor_final.ckpt --config /assets/model_config.yaml \
  --detector /assets/detector.pt --export-baseline-crops
python scripts/extract_dino_features.py --data "$DATA" \
  --crops "$DATA/crops" --checkpoint /assets/dinov2_vitb14.pth
```

If WiLoR features already exist, use `--overwrite` to also create missing
crops. This consumes additional disk space. The crop exporter saves
`crops/<trial_id>/rgb.npz` (`img`, `right`, `box_center`, `box_size`, `img_size`)
and `dino.npy` (224×224 ImageNet-normalized RGB). Native RGB crops are 256×256,
ImageNet-normalized, mirrored for a left hand using the official ViTDet crop
pipeline. No GT geometry or region-force label defines the crop. An ambiguous
or missing detector result aborts extraction for review rather than silently
inventing a hand or filling features with zeros. Keep extraction receipts.

Compute `normalization-<protocol>.json` from TRAIN for each protocol as in the
root README. It is required for MoT/HBridge; vision-only methods ignore EMG.

## 2. Acquire external components/assets

For NeuroPose/EMGFormer/vemg2pose, first read the upstream licenses, then run:

```bash
python -m baselines.prepare_sources --out /assets/egomyo-components \
  --accept-upstream-licenses
```

This downloads source only, with a receipt and license/notices files. EgoEMG
revision: `e2d1fa5ed793c67abc23dbfa2d8fe309aaf61d7b`; Meta emg2pose revision:
`5f6f62b1a0a08426adffe55900842e75a8adb38c`. The four EgoEMG components are also
byte-pinned against the research source snapshot. Preserve the receipt and
review mixed third-party terms (including Meta CC-BY-NC-SA-4.0). External code
is executed only after verification; do not use untrusted source directories.
Example `emg-assets.json`:

```json
{"components": "/assets/egomyo-components"}
```

For RGB methods, install the appropriate upstream environment separately:
[HaMeR](https://github.com/geopavlakos/hamer),
[WiLoR](https://github.com/rolpotamkin/WiLoR). Do not redistribute their
weights or licensed MANO. Record/pin the checkout commit and install the
dependencies required by that checkout. Example `wilor-assets.json`:

```json
{
  "source_root": "/opt/WiLoR",
  "checkpoint": "/assets/wilor_final.ckpt",
  "config": "/assets/model_config.yaml"
}
```

For HaMeR, use the same keys without `config`; its official loader resolves
the configuration adjacent to its checkpoint. These RGB recipes initialize
the native model from the publisher checkpoint, attach fresh physical heads,
and train the learned native/task parameters. MANO asset parameters are frozen
and omitted from saved release checkpoints. RGB training may require a much
smaller batch (start with 2–8); it does not run from the 13 cached WiLoR tokens.
Only MyoBridge uses that frozen WiLoR token input.

## 3. Train and evaluate

Run from the repository root. Example for a cached visual baseline:

```bash
python -m baselines.run train --method egopressure --data "$DATA" \
  --splits "$DATA/splits.json" --protocol cross_trial \
  --partition "$DATA/anatomical_partition.npz" --seed 0 --epochs 20 \
  --batch 128 --workers 4 --out runs/egopressure/cross_trial/seed0
python -m baselines.run evaluate --method egopressure --data "$DATA" \
  --splits "$DATA/splits.json" --protocol cross_trial \
  --partition "$DATA/anatomical_partition.npz" --seed 0 --epochs 20 \
  --mano /assets/mano_buffers.npz \
  --checkpoint runs/egopressure/cross_trial/seed0/final.pt \
  --out runs/egopressure/cross_trial/seed0/test.json
```

For `mot`/`hbridge` add
`--normalization "$DATA/normalization-cross_trial.json"` to **both** commands.
For `neuropose`/`emgformer`/`emg2pose` add `--assets emg-assets.json` to both.
For `hamer`/`wilor` pass its RGB asset configuration and an appropriate batch.
Repeat for `cross_session` and `cross_subject`, seeds `0`, `1`, `2`, keeping
the split fixed across model seeds. Use a distinct output directory for each
run. These commands do not allocate GPUs or start cluster jobs automatically.

The shared release recipe uses AdamW (`lr=5e-4`, weight decay `1e-4`), two-epoch
warmup/cosine decay, gradient clipping at 5, and 20 fixed epochs. The objective
averages active masked geometry/contact/force heads: geometry L1 ×100,
contact BCE, estimated-force L1; HBridge adds masked semantic reconstruction.
No TEST access occurs during training, early stopping or checkpoint selection.
Evaluation loads only the specified final epoch, validates data/split/asset
provenance and uses the same `myobridge.metrics.Metrics` as MyoBridge.
Metric definitions and target fields are in
[`../docs/data-field-mapping.md`](../docs/data-field-mapping.md).

Metrics include validity coverage; compare methods on common annotation/input
support and report those counts, especially when a 2000-sample history is needed.

The included CPU tests cover cached-model forwards/backprop, prediction-only
inputs, long-history masks and the fixed-epoch runner. Licensed RGB networks
require an installed upstream environment and assets; synthetic interface tests
are not a substitute for running those external models on approved recordings.
To also exercise the three real external EMG component graphs, run
`EGOMYO_COMPONENTS=/assets/egomyo-components python -m pytest tests/test_baselines.py`.

# Raw recordings to training targets

[Dataset](https://huggingface.co/datasets/long2333/EgoMyo) ·
[Code](https://github.com/SunGL001/EgoMyo) · [Website](https://egomyo.github.io/)

`scripts/prepare_targets.py` reads delivered sensor **and annotation** files and
creates the model-ready layout used by MyoBridge and the baseline recipes.
**pressure2 is the default.** It uses completed, unsmoothed regional force;
it does not silently substitute pressure1 or apply an additional smoothing pass.
The arXiv link will be added when available.

## 1. Required inputs

Use the approved release membership JSON, not a directory listing or an inferred
participant identity. Every row under `trials` needs `trial_id`, `source_run_id`,
`relative_path`, `package`, `side` (`L`/`R`), `session` (1/2), `action`, and
`repetition` (1–5). To generate cross-subject splits it also needs the reviewed
`participant` ID: both hands and all sessions of a person share that ID.
`participant_candidate` is not automatically promoted to an authenticated ID.
An optional `inputs` map records `{relative_filename: {bytes, sha256}}`; present
hashes are verified before conversion. Do not edit frozen hashes to hide a
changed source file.

Under each row's `relative_path`, the converter requires:

| Source | Contents/use |
| --- | --- |
| `trial_info.json` | Trial, run, package, session/action/repetition identity. |
| `ego/ego_timestamps.csv` | `host_receive_timestamp_ns`, consecutive zero-based `output_frame_index`, trial/run identity. Defines the output observation timeline. |
| `emg/emg_raw.h5` | `adc[N,18]`, `host_monotonic_ns`, `sample_index`: causal EMG input. |
| `contact/mano.npz` | Original `vertices_ego_m[F,778,3]`, `joints_3d_ego_m[F,21,3]`, `faces[1538,3]`, timestamps, trial/run IDs, `mesh_valid`, `original_valid`, `ego_coordinate_status`; optional `source_timing_rejected`. Geometry supervision, not network inputs. |
| `force/pressure_completed.h5` | pressure2 timeline, unsmoothed `force_7_estimated_n`, quality/source-row provenance. Regional-force supervision. |
| `force/region_force_completed.csv` | Independent completed timeline/regional-force/quality cross-check. |
| `force/pressure_completion.json` | Completion version and trial/run identity. |
| `contact/contact_surface_completed.h5` | Completed per-face matrix force, region mapping, spatial-mapping validity, and matching pressure2 timeline. Surface supervision. |
| `contact/contact_postprocessed.h5` | Original anatomical region partition/hash; also the pressure1 surface source if explicitly selected. |
| `force/pressure_postprocessed.h5`, `force/region_force_postprocessed.csv`, `force/pressure_postprocess.json` | Original pressure source and provenance; retain these alongside pressure2. The completed H5 can contain a virtual matrix referencing the original H5. |

The source root also contains `<package>/DATASET.json`. Raw video is
`ego/ego.mp4` and is consumed by the visual-feature extraction step, not the
sensor/annotation converter. Sensor-only ADC/pressure matrices without delivered
MANO, calibration/region mapping and completion annotations are insufficient:
this tool does not invent those missing annotations or run the acquisition UI.

## 2. Convert without changing the source

```bash
python scripts/prepare_targets.py \
  --source-root /data/EgoMyo/raw \
  --membership /data/EgoMyo/release-membership.json \
  --out /data/EgoMyo/prepared-pressure2 \
  --include-long-emg
```

The output directory must not exist. `--include-long-emg` additionally exports
`emg_2000` and `input_valid_2000` for NeuroPose, EMGFormer and vemg2pose; MyoBridge
still uses `emg[18,400]`. Omit it if those baselines are not needed.
`--pressure-version pressure1` is an explicit optional alternate annotation
source; it never becomes an automatic fallback for a missing pressure2 file.

Outputs: `targets/<trial_id>.npz`, `anatomical_partition.npz`, `membership.json`
with frame counts, `frame-map.json`, per-trial SHA-256 receipts and coverage in
`provenance/`, and a final `preparation.json` recording the conversion policy.
Only an export containing `preparation.json` with `complete: true` is complete;
delete or move a failed partial output before retrying into a new directory.

## 3. Alignment, labels and masks

- **EMG:** at each video timestamp, take only samples already received from that
  trial. Use the last 400 samples, 18 channels, channel-first; never read future
  samples or neighboring trials. Padding, sample-index gaps, receive gaps over
  50 ms, a latest sample over 20 ms old, or a window starting over 250 ms before
  the observation mark the window invalid. The optional 2000-sample history
  uses a 1100 ms maximum age instead. No filtering or electrode reordering is
  added: ADC channel order is preserved.
- **Pose:** nearest original annotation within 50 ms, earlier on ties. Subtract
  the original wrist (joint 0), retaining metres and the original joint order.
  Require original mesh/wrist validity, finite coordinates, accepted frozen
  coordinate status and no explicit timing rejection. Unknown timing remains
  recorded as `timing_unknown`; it is not a claim of verified physical sync.
  Completed preview meshes are not substituted for original pose labels.
- **pressure2 force:** nearest completed row within 40 ms, earlier on ties.
  Region order is `T, I, M, R, P, PALM_UPPER, PALM_LOWER`. Reject unresolved
  (`8`) and missing-source (`64`) flags, nonfinite/negative values and invalid
  EMG. Interpolated, inserted and estimated rows retain their original flags
  and source row (`-1` for insertions); they are derived labels, not measured
  force. `pressure_source_quality` preserves the original pressure flags.
- **pressure2 contact:** completed CSV has no explicit contact-state columns.
  Define regional contact as completed unsmoothed force **strictly greater than
  0.2 estimated N** (`--contact-threshold-n`); validity follows force validity.
  Invalid rows have unknown global state `-1`, not valid no-contact labels.
- **Surface:** require an available completed spatial mapping, valid original
  mesh, valid pressure/EMG and known contact. Assign each vertex to the majority
  region of its incident faces (ties: lower region ID; unassigned: `-1`).
  Distribute each completed face's force equally among its owner-matching
  corners, renormalizing those weights so supported force is conserved. Missing
  owner support invalidates the distribution. No seven-region force is
  broadcast onto the whole mesh. `surface_regional_force` sums this mapped
  surface force; it is not interchangeable with `regional_force`. Vertex
  contact requires positive mapped force and a contact-positive owner region.
  Unknown mapping is masked rather than fabricated.

The optional pressure1 branch uses delivered explicit regional/global contact,
original sparse barycentric mapping and its own pressure-quality semantics.
`preparation.json` identifies the selected policy; do not mix annotation versions
within one reported comparison.

## 4. Extract features, freeze splits and train

```bash
DATA=/data/EgoMyo/prepared-pressure2
python scripts/extract_wilor_from_video.py --data "$DATA" \
  --frame-map "$DATA/frame-map.json" --wilor-root /opt/WiLoR \
  --checkpoint /assets/wilor_final.ckpt --config /assets/model_config.yaml \
  --detector /assets/detector.pt
python scripts/make_splits.py --membership "$DATA/membership.json" \
  --split-seed 0 --out "$DATA/splits.json"
python scripts/compute_normalization.py --data "$DATA" --splits "$DATA/splits.json" \
  --protocol cross_trial --out "$DATA/normalization-cross_trial.json"
python scripts/validate_dataset.py --data "$DATA" \
  --normalization "$DATA/normalization-cross_trial.json"
```

Prepare your licensed MANO buffers with `scripts/prepare_mano.py`, then follow
the root README training/evaluation commands, passing
`--normalization "$DATA/normalization-cross_trial.json"`. Compute separate
TRAIN-only statistics for each protocol. Use the same frozen split for all
model seeds and methods. See [`../baselines/README.md`](../baselines/README.md)
for baseline-specific inputs and entry points.

Source acceptance and `cache_release_allowed` flags are copied into the export,
not overridden. A local conversion is not permission to redistribute data.
Do not upload source recordings, derived caches or licensed model assets until
their release and participant-consent checks are approved. Force remains an
uncalibrated estimate, and imputation is never relabeled as a measurement.

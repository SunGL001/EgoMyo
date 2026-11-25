# EgoMyo dataset card — draft template

> **Draft only. Do not publish this file as the dataset card until all TODOs,
> data-rights reviews and release-eligibility checks are complete.** Do not
> infer a license or dataset statistics from the model-code repository.

## Dataset summary

- **Dataset name:** EgoMyo
- **Version/revision:** TODO — set from the approved release manifest
- **Maintainer/contact:** TODO
- **License:** TODO — add only the license approved for the data and annotations
- **Access:** TODO — public, gated or private; document request requirements
- **Related project:** [Project website](https://egomyo.github.io/)
- **Code:** [SunGL001/EgoMyo](https://github.com/SunGL001/EgoMyo)
- **Paper:** arXiv link pending

EgoMyo is a multimodal hand-interaction dataset containing egocentric video,
wrist surface electromyography (sEMG), and hand pose/contact/force annotations.
Replace this summary with a concise description of the exact released subset.

## Intended use

TODO — describe supported research tasks, permitted use, prohibited use and
whether the release may be used for commercial, biometric or safety-critical
applications. Do not imply participant consent covers uses that were not
approved.

## Collection and annotation

TODO — document participants and sessions using approved non-identifying
statistics; acquisition hardware/configuration; frame and EMG rates; timing and
synchronization; pose/contact/force annotation methods; quality control;
calibration; and known gaps. State explicitly whether force values are
calibrated measurements or estimates.

## Files and schema

TODO — list the exact versioned directory structure and formats in the release,
including a table of fields, shape, dtype, units, coordinate frame, missing
value representation and validity mask for each modality. Include one synthetic
example with no participant data. Identify which files are source recordings
and which are processed/model-ready exports.

For the MyoBridge training-ready export, see the expected layout and tensor
shapes in the code repository's [README](https://github.com/SunGL001/EgoMyo).
The training loader does not automatically convert an acquisition/source
package into that export.

## Splits and evaluation

TODO — publish the split-generation version and exact protocol definitions.
Explain the grouping unit and demonstrate that participants do not leak across
the cross-subject split. State whether any public benchmark results used the
released test labels.

## Privacy, consent and limitations

TODO — summarize the approved consent scope, privacy review, de-identification
steps, residual re-identification risks, known missingness and quality issues,
and a contact for data-subject requests. Confirm that video, sensor streams and
derived files were all reviewed; removing names from metadata alone is not
de-identification.

## Licensing and third-party assets

TODO — specify the data license and attribution requirements after approval.
List separately any third-party assets and their licenses. Do not bundle
licensed MANO, WiLoR or DINOv2 model weights unless redistribution is expressly
permitted.

## Download and citation

TODO — add the approved download/access instructions and the citation after the
paper/preprint exists. The arXiv link will be added when available.

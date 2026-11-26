# EgoMyo Dataset Guide

The dataset card is maintained with the data on
[Hugging Face](https://huggingface.co/datasets/long2333/EgoMyo).
It describes the recordings, pressure2 annotations, archive layout,
fixed cross-trial split, limitations and download instructions.

| Resource | Link |
| --- | --- |
| Paper | [EgoMyo on arXiv](https://arxiv.org/abs/2611.34224) |
| Dataset | [long2333/EgoMyo](https://huggingface.co/datasets/long2333/EgoMyo) |
| Code | [SunGL001/EgoMyo](https://github.com/SunGL001/EgoMyo) |
| Website | [egomyo.github.io](https://egomyo.github.io/) |

## Using the Dataset with This Code

1. Download the desired archives and metadata with `scripts/download_dataset.py`.
2. Verify archive checksums and extract the packages into one raw root.
3. Run `scripts/prepare_targets.py` to align EMG, pose and pressure2 annotations.
4. Extract visual features from video and compute normalization from TRAIN only.
5. Train and evaluate using the frozen splits and validity masks.

Follow the [README quick start](../README.md) or the detailed
[raw-to-target workflow](raw-to-targets.md). The
[field mapping](data-field-mapping.md) specifies which data are model inputs,
which are supervision, tensor shapes, units and masking rules.

The public dataset includes corrected/completed **pressure2 only**; it does not
require the earlier pressure files. Missing or estimated annotations retain
their validity and provenance flags. Force values are estimates rather than
independently calibrated measurements. Licensed MANO assets and external model
weights must be obtained separately under their respective terms.

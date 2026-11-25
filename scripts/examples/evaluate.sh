#!/usr/bin/env bash
# Score the final-epoch weights on the held-out split.
set -euo pipefail
DATA=${DATA:?set DATA to the dataset root}
SPLITS=${SPLITS:-$DATA/splits.json}
NORMALIZATION=${NORMALIZATION:-$DATA/normalization.json}
MANO=${MANO:?set MANO to the MANO buffer archive}
PARTITION=${PARTITION:?set PARTITION to anatomical_partition.npz}
RUN=${RUN:?set RUN to a training run directory}

python scripts/evaluate.py \
  --data "$DATA" --splits "$SPLITS" --protocol cross_trial \
  --normalization "$NORMALIZATION" \
  --mano "$MANO" --partition "$PARTITION" \
  --checkpoint "$RUN/checkpoint.pt" --visual wilor --seed 0 \
  --out "$RUN/test_metrics.json"

#!/usr/bin/env bash
# Default configuration: frozen WiLoR tokens, no pose or shape prior.
set -euo pipefail
DATA=${DATA:?set DATA to the dataset root}
SPLITS=${SPLITS:-$DATA/splits.json}
NORMALIZATION=${NORMALIZATION:-$DATA/normalization.json}
MANO=${MANO:?set MANO to the MANO buffer archive}
PARTITION=${PARTITION:?set PARTITION to anatomical_partition.npz}
OUT=${OUT:-runs/cross_trial_wilor_s0}

python scripts/train.py \
  --data "$DATA" --splits "$SPLITS" --protocol cross_trial \
  --normalization "$NORMALIZATION" \
  --mano "$MANO" --partition "$PARTITION" --visual wilor \
  --seed 0 --epochs 20 --batch 256 --block 128 --lr 5e-4 --out "$OUT"

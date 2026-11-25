#!/usr/bin/env bash
# Optional variant: cached DINOv2 features through a residual adapter.
set -euo pipefail
DATA=${DATA:?set DATA to the dataset root}
SPLITS=${SPLITS:-$DATA/splits.json}
MANO=${MANO:?set MANO to the MANO buffer archive}
PARTITION=${PARTITION:?set PARTITION to anatomical_partition.npz}
OUT=${OUT:-runs/cross_trial_dino_s0}

python scripts/train.py \
  --data "$DATA" --splits "$SPLITS" --protocol cross_trial \
  --mano "$MANO" --partition "$PARTITION" --visual dino \
  --seed 0 --epochs 20 --batch 256 --block 128 --lr 5e-4 --adapter-lr 5e-5 --out "$OUT"

#!/usr/bin/env bash
# Replicate the 256-step all_future run with a second training seed (43).
#
# Same data, same hyperparameters and same protocol as af256_s42 -- only the model
# init seed and the three RNG streams shift. The point is to separate "the gain is
# real" from "seed 42 happened to be lucky", which the paired prompt bootstrap
# cannot answer on its own.
#
# Usage: bash scripts/train_seed43.sh
set -eu

ROOT=/physis/unified-dopsd/mmada-opsd-clean
PY=/physis/unified-dopsd/venv/bin/python
PROMPTS=/physis/unified-dopsd/full_future_h800/train256_parti_prompts.json
OUT=$ROOT/runs/af256_s43

export PYTHONPATH="$ROOT"
cd "$ROOT"
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 "$PY" -m opsd.train \
  --teacher all_future \
  --endpoint-mode visible \
  --gap 2 \
  --student-selection fixed \
  --mask-threshold 0.75 \
  --loss forward_kl \
  --updates 256 \
  --steps 15 \
  --cfg 3.5 \
  --anneal upstream \
  --lr 2e-5 \
  --grad-clip 1.0 \
  --seed 43 \
  --rollout-seed 2000 \
  --selection-seed 400000 \
  --prompts "$PROMPTS" \
  --out "$OUT" \
  > "$OUT/train.log" 2>&1

echo "training done: $(cat "$OUT/TRAIN_DONE.json" 2>/dev/null)"

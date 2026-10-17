#!/usr/bin/env bash
# Re-run the remaining teacher-view arms under the clean upstream protocol:
#
#   endpoint_visible   -- teacher sees the final decoded image as-is
#   endpoint_remasked  -- final image with the student's positions re-masked
#   gap_n (gap=2)      -- teacher sees the state 2 steps ahead
#
# all_future was validated at +4.10pp [+2.88, +5.31]; these arms reuse the exact
# seed-42 protocol of af256_s42 so every number is directly comparable. Phases:
# train all three in parallel (one GPU each), then per arm generate 1600 images
# (8 shards) and score (2 seeds x 8 shards).
#
# Usage: bash scripts/run_other_arms.sh
set -u

ROOT=/physis/unified-dopsd/mmada-opsd-clean
PY=/physis/unified-dopsd/venv/bin/python
PROMPTS=/physis/unified-dopsd/full_future_h800/train256_parti_prompts.json
DATA=/physis/unified-dopsd/geneval2_data.jsonl
MODEL_DIR=/physis/models/Qwen3-VL-8B-Instruct
SEEDS="17000 29000"

export PYTHONPATH="$ROOT"
cd "$ROOT"

# Note: --endpoint-mode is deliberately not passed for the two endpoint arms. The
# view itself selects visible-vs-remasked (see build_target in opsd/teacher.py), so
# passing the flag there would look meaningful while doing nothing.
declare -A ARM_ARGS=(
  [ev256_s42]="--teacher endpoint_visible"
  [er256_s42]="--teacher endpoint_remasked"
  [g2256_s42]="--teacher gap_n --gap 2"
)
ARMS=(ev256_s42 er256_s42 g2256_s42)

echo "=== phase 1: training 3 arms in parallel ==="
gpu=0
for arm in "${ARMS[@]}"; do
  mkdir -p "runs/$arm"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -m opsd.train \
    ${ARM_ARGS[$arm]} \
    --student-selection fixed --mask-threshold 0.75 --loss forward_kl \
    --updates 256 --steps 15 --cfg 3.5 --anneal upstream \
    --lr 2e-5 --grad-clip 1.0 \
    --seed 42 --rollout-seed 1000 --selection-seed 300000 \
    --prompts "$PROMPTS" --out "runs/$arm" \
    > "runs/$arm/train.log" 2>&1 &
  gpu=$((gpu + 1))
done
wait

for arm in "${ARMS[@]}"; do
  if [ ! -f "runs/$arm/TRAIN_DONE.json" ]; then
    echo "ERROR: $arm training did not finish"; exit 1
  fi
  echo "$arm: $(cat runs/$arm/TRAIN_DONE.json)"
done

echo "=== phase 2+3: generate and score each arm ==="
for arm in "${ARMS[@]}"; do
  out="runs/$arm/geneval2_imgs"
  mkdir -p "$out"
  echo "=== generating $arm (8 shards) ==="
  for s in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$s "$PY" -m opsd.generate \
      --checkpoint "runs/$arm/checkpoint-final" \
      --prompts "$DATA" --seeds $SEEDS --out "$out" \
      --shard-index "$s" --num-shards 8 --steps 15 --cfg 3.5 \
      > "runs/$arm/gen_shard${s}.log" 2>&1 &
  done
  wait
  n=$(find "$out" -name '[0-9]*.png' | wc -l)
  echo "$arm: $n images"
  [ "$n" -eq 1600 ] || { echo "ERROR: $arm expected 1600 got $n"; exit 1; }

  pack=/physis/unified-dopsd/gv2_scores/$arm
  "$PY" -m eval.pack_for_geneval2 --images "$out" --data "$DATA" \
    --out-dir "$pack" --num-seeds 2
  for seed_idx in 0 1; do
    d="$pack/seed${seed_idx}"
    echo "=== scoring $arm seed${seed_idx} ==="
    for s in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$s "$PY" -m eval.score_geneval2 \
        --benchmark "$d/benchmark.jsonl" --images "$d/image_paths.json" \
        --out "$d/shard${s}.json" --model "$MODEL_DIR" \
        --method soft_tifa_am --shard-index "$s" --num-shards 8 \
        > "$d/shard${s}.log" 2>&1 &
    done
    wait
    for m in soft_tifa_am soft_tifa_gm; do
      "$PY" -m eval.score_geneval2 --merge --out-dir "$d" --shards 8 \
        --method $m 2>&1 | tee "$d/merged_$m.txt"
    done
  done

  echo "=== compare $arm vs base ==="
  "$PY" -m eval.compare --root /physis/unified-dopsd/gv2_scores \
    --data "$DATA" --trained "$arm" --base base || true
done

echo "ALL ARMS DONE"

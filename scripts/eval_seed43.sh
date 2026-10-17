#!/usr/bin/env bash
# Generate images and run GenEval2 scoring for seed43-trained checkpoint.
#
# Layout mirrors gen_geneval2.sh: 800 prompts × 2 image seeds × 1 model = 1600 images,
# split across GPUs 0-3 (4 shards). Then pack and score with 8-shard Qwen3-VL.
#
# Usage: bash scripts/eval_seed43.sh
set -eu

ROOT=/physis/unified-dopsd/mmada-opsd-clean
PY=/physis/unified-dopsd/venv/bin/python
DATA=/physis/unified-dopsd/geneval2_data.jsonl
CKPT=$ROOT/runs/af256_s43/checkpoint-final
SEEDS="17000 29000"
SHARDS=4

export PYTHONPATH="$ROOT"

if [ ! -f "$CKPT/adapter_model.safetensors" ]; then
  echo "ERROR: checkpoint not found at $CKPT"
  exit 1
fi

OUT="$ROOT/runs/af256_s43/geneval2_imgs"
mkdir -p "$OUT"

echo "=== generating 1600 images on GPUs 0-3 ==="
for ((s=0; s<SHARDS; s++)); do
  CUDA_VISIBLE_DEVICES=$s "$PY" -m opsd.generate \
    --checkpoint "$CKPT" \
    --prompts "$DATA" \
    --seeds $SEEDS \
    --out "$OUT" \
    --shard-index $s --num-shards $SHARDS \
    --steps 15 --cfg 3.5 \
    > "$ROOT/runs/af256_s43/gen_shard${s}.log" 2>&1 &
done
wait

ACTUAL=$(find "$OUT" -name '[0-9]*.png' 2>/dev/null | wc -l)
echo "$ACTUAL images written"
[ "$ACTUAL" -eq 1600 ] || { echo "ERROR: expected 1600, got $ACTUAL"; exit 1; }

echo "=== packing for GenEval2 scorer ==="
PACK_ROOT=/physis/unified-dopsd/gv2_scores/af256_s43
"$PY" -m eval.pack_for_geneval2 \
  --images "$OUT" \
  --data "$DATA" \
  --out-dir "$PACK_ROOT" \
  --num-seeds 2

echo "=== scoring seed43 (2 seeds × 8 shards) ==="
cd "$ROOT"
MODEL_DIR=/physis/models/Qwen3-VL-8B-Instruct
for seed_idx in 0 1; do
  d="$PACK_ROOT/seed${seed_idx}"
  echo "=== af256_s43 seed${seed_idx}: 8 shards ==="
  for shard in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$shard "$PY" -m eval.score_geneval2 \
      --benchmark "$d/benchmark.jsonl" \
      --images "$d/image_paths.json" \
      --out "$d/shard${shard}.json" \
      --model "$MODEL_DIR" \
      --method soft_tifa_am \
      --shard-index $shard --num-shards 8 \
      > "$d/shard${shard}.log" 2>&1 &
  done
  wait
  for m in soft_tifa_am soft_tifa_gm; do
    "$PY" -m eval.score_geneval2 --merge --out-dir "$d" --shards 8 \
      --method $m 2>&1 | tee "$d/merged_$m.txt"
  done
done

echo "=== comparing seed43 vs base ==="
"$PY" -m eval.compare --root /physis/unified-dopsd/gv2_scores \
  --data "$DATA" --trained af256_s43 --base base

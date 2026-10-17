#!/usr/bin/env bash
# Score GenEval2 for both checkpoints, both seeds, sharded across GPUs.
#
#   2 models x 2 seeds x 800 prompts x ~7.5 VQA forwards = 24000 single-token
#   forwards. One shard per GPU; 8 shards run at once.
#
# Note the two-level layout: pack_for_geneval2 takes an out-dir and itself writes
# seed0/ and seed1/ inside it, so packing goes to $WORK/<model> and scoring reads
# $WORK/<model>/seed<N>.
#
# Usage: bash scripts/score_geneval2.sh [num_shards]
set -u

ROOT=/physis/unified-dopsd/mmada-opsd-clean
PY=/physis/unified-dopsd/venv/bin/python
DATA=/physis/unified-dopsd/geneval2_data.jsonl
WORK=/physis/unified-dopsd/gv2_scores
SHARDS=${1:-8}
MODEL_DIR=/physis/models/Qwen3-VL-8B-Instruct

export PYTHONPATH="$ROOT"
mkdir -p "$WORK"

for model in af256_s42 base; do
  imgs="$ROOT/runs/$model/geneval2_imgs"
  echo "=== packing $model ==="
  "$PY" -m eval.pack_for_geneval2 \
    --images "$imgs" --data "$DATA" --out-dir "$WORK/$model" --num-seeds 2 || exit 1

  for seed in 0 1; do
    d="$WORK/$model/seed$seed"
    echo "=== $model seed$seed: $SHARDS shards ==="
    for ((s=0; s<SHARDS; s++)); do
      CUDA_VISIBLE_DEVICES=$s "$PY" -m eval.score_geneval2 \
        --benchmark "$d/benchmark.jsonl" \
        --images "$d/image_paths.json" \
        --out "$d/shard${s}.json" \
        --model "$MODEL_DIR" \
        --method soft_tifa_am \
        --shard-index $s --num-shards $SHARDS \
        > "$d/shard${s}.log" 2>&1 &
    done
    wait
    for m in soft_tifa_am soft_tifa_gm; do
      "$PY" -m eval.score_geneval2 --merge --out-dir "$d" --shards $SHARDS \
        --method $m 2>&1 | tee "$d/merged_$m.txt"
    done
  done
done

echo "=== all scores ==="
for model in af256_s42 base; do
  for seed in 0 1; do
    f="$WORK/$model/seed$seed/score.json"
    [ -f "$f" ] && echo "$model seed$seed: $(cat "$f")"
  done
done

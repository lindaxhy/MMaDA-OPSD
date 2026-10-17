#!/usr/bin/env bash
# Generate GenEval2 images for base and trained checkpoints, sharded across GPUs.
#
#   800 prompts x 2 seeds x 2 models = 3200 images
#   traced at ~6 s/image -> a single model over 4 GPUs is ~40 min
#
# Each model is split across the GPUs given to it, so shard count == GPU count.
#
# Usage: bash scripts/gen_geneval2.sh [shards_per_model]
set -u

ROOT=/physis/unified-dopsd/mmada-opsd-clean
PY=/physis/unified-dopsd/venv/bin/python
DATA=/physis/unified-dopsd/geneval2_data.jsonl
SEEDS="17000 29000"
SHARDS=${1:-4}

export PYTHONPATH="$ROOT"

# args: checkpoint tag first_gpu n_gpus
run_model () {
  local ckpt="$1" tag="$2" gpu_base="$3" ngpu="$4"
  local out="$ROOT/runs/$tag/geneval2_imgs"
  mkdir -p "$out"
  echo "=== $tag: $ngpu shards on GPU $gpu_base..$((gpu_base+ngpu-1)) -> $out ==="
  for ((s=0; s<ngpu; s++)); do
    CUDA_VISIBLE_DEVICES=$(( gpu_base + s )) "$PY" -m opsd.generate \
      --checkpoint "$ckpt" \
      --prompts "$DATA" \
      --seeds $SEEDS \
      --out "$out" \
      --shard-index $s --num-shards $ngpu \
      --steps 15 --cfg 3.5 \
      > "$ROOT/runs/$tag/gen_shard${s}.log" 2>&1 &
  done
}

# trained on GPUs 0-3, base on GPUs 4-7
run_model "$ROOT/runs/af256_s42/checkpoint-final" "af256_s42" 0 4
run_model "base" "base" 4 4
wait

echo "=== done ==="
for t in af256_s42 base; do
  echo "$t: $(find "$ROOT/runs/$t/geneval2_imgs" -name '[0-9]*.png' 2>/dev/null | wc -l) images"
done

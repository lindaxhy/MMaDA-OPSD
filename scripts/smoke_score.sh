#!/usr/bin/env bash
# Smoke-test the GenEval2 scoring path end to end before the full 32-shard run.
# Loads Qwen3-VL once on one GPU, scores 5 prompts, and prints the aggregate.
set -eu

ROOT=/physis/unified-dopsd/mmada-opsd-clean
PY=/physis/unified-dopsd/venv/bin/python
DATA=/physis/unified-dopsd/geneval2_data.jsonl
TMP=/physis/unified-dopsd/gv2_smoke
export PYTHONPATH="$ROOT"

rm -rf "$TMP"; mkdir -p "$TMP"

# pack 5 prompts (limit applies to the benchmark, so both seeds get 5)
"$PY" - "$DATA" "$TMP/bench5.jsonl" <<'PYEOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(src)][:5]
with open(dst, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} prompts to {dst}")
PYEOF

"$PY" -m eval.pack_for_geneval2 --images "$ROOT/runs/base/geneval2_imgs" \
  --data "$TMP/bench5.jsonl" --out-dir "$TMP/packed" --num-seeds 2 >/dev/null

echo "--- probing 1 prompt on GPU 0 ---"
CUDA_VISIBLE_DEVICES=0 timeout 900 "$PY" - "$TMP/packed/seed0/benchmark.jsonl" \
  "$TMP/packed/seed0/image_paths.json" "$ROOT" <<'PYEOF'
import json, sys, time
bench, imgs, root = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root)
from eval.score_geneval2 import load_scorer, aggregate

entries = [json.loads(l) for l in open(bench)][:1]
image_data = json.loads(open(imgs).read())
t0 = time.time()
ev = load_scorer("/physis/models/Qwen3-VL-8B-Instruct")
print(f"model loaded on {ev.qwen_model.device} in {time.time()-t0:.1f}s")

# exercise both the raw VQA call and the tifa loop
p = entries[0]
path = image_data[p["prompt"]]
pred, prob = ev.send_message_with_image("Does this image show a green backpack? Answer in one word.", path, answer_list=["Yes", "yes", " yes", " Yes"])
print(f"raw VQA -> pred={pred!r} prob_yes={prob:.4f}")

t1 = time.time()
score, per = ev.soft_tifa(p["vqa_list"], path)
dt = time.time() - t1
print(f"soft_tifa on {len(p['vqa_list'])} questions -> {score:.3f} in {dt:.1f}s ({dt/len(p['vqa_list']):.1f}s/question)")
print(f"per-question: {[round(x,3) for x in per]}")
PYEOF

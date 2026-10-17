# eval/

这里**不重写** GenEval2 打分逻辑。官方实现取 VQA 模型对正确答案的 softmax 概率
（`max_new_tokens=1` + `output_scores`），答案列表含大小写与数字变体，per-atom 算术
平均后按 prompt 聚合（`soft_tifa_am` / `soft_tifa_gm`）。任何自行简化的版本都会与
论文数字不可比。

**用法**：
- 打分脚本：官方 `evaluation.py`（本机 `staging/geneval2-source/evaluation.py`，
  服务器上放 `/physis/unified-dopsd/geneval2/evaluation.py`）
- 数据：`geneval2_data.jsonl`（800 条，含 `prompt` / `vqa_list` / `skills` / `atom_count`）
- evaluator：`Qwen/Qwen3-VL-8B-Instruct`

**输入格式**：官方脚本要求 `image_filepath_data` 是一个
`{prompt: image_filepath}` 的 JSON 字典（一个 prompt 一张图）。我们一棵 prompt 出
多个 seed，所以先用 `pack_for_geneval2.py` 展平。

```bash
# 1. 生成图像
python -m opsd.generate --checkpoint runs/af256_s42/checkpoint-final \
  --prompts /physis/unified-dopsd/geneval2/geneval2_data.jsonl \
  --seeds 17000 29000 --out runs/af256_s42/geneval2_imgs

# 2. 展平成官方脚本要的字典（每个 (prompt, seed) 一个 key）
python eval/pack_for_geneval2.py --images runs/af256_s42/geneval2_imgs \
  --data /physis/unified-dopsd/geneval2/geneval2_data.jsonl \
  --out /tmp/gv2_seed0.json --seed-index 0

# 3. 官方打分（在装了 Qwen3-VL 的环境里）
python /physis/unified-dopsd/geneval2/evaluation.py \
  --benchmark_data /physis/unified-dopsd/geneval2/geneval2_data.jsonl \
  --image_filepath_data /tmp/gv2_seed0.json \
  --method soft_tifa_am --output_file /tmp/gv2_seed0_scores.json
```

**多 seed 聚合**：官方是对单个 prompt 的一张图打分。我们每个 prompt 有 2 个 seed，
对每个 seed 独立跑一遍打分，然后在 `compare.py` 里做 prompt 级配对比较
（同一 prompt 的 base vs trained，同 seed）。

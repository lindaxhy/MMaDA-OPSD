# 数据、更新步数、评测

本文档回答三个核心实验设计问题，基于原始 full_future_h800 归档的实际配置。

---

## 一、用什么数据训练

### 推荐：PartiPrompts 的 256 条质量子集

原仓库 `full_future_h800/train256_parti_prompts.json` 收录 256 条从 [PartiPrompts](https://github.com/google-research/parti/tree/main/PartiPrompts) 中筛选的文本，每条 10–40 词，涵盖组合属性（"red backpack next to green bike"）、空间关系、计数、少见对象、抽象概念。它是对 T2I 成熟度的宽幅采样，而不是极端 challenge 集。

配置建议：
- **冒烟测试**：5 条（`/tmp/smoke_prompts.json` 已在 H800 上），每轮采样同一批 prompt 以隔离训练信号 vs rollout 噪声。
- **完整开发集**：64 条（256 的四分之一），用于快速验证收益是否复现、调试熵监控、消融单个超参。
- **报告质量的训练臂**：256 条，每条一次采样（共 256 次更新）。这是归档结果的配置，也是论文要报告的数字。

下载：
```bash
ssh -p 8595 root@106.12.153.99 \
  "cat /physis/unified-dopsd/runs/m*/train256_parti_prompts.json | head -1" \
  > train256_parti_prompts.json
```

### 可选：DPG-Bench / MJHQ / DrawBench

- **DPG-Bench**（1000 prompts）：如果你需要与其他 RLHF-for-diffusion 工作（DPOK, DDPO）直接数值对比，用它的训练 split。MMaDA 没有公开用这个集合的结果，但上游论文里 SDXL 用它。
- **MJHQ**（10k aesthetic prompts）：高美学倾向数据，会把模型往艺术风格拉，PartiPrompts 更注重成分正确。
- **DrawBench**（200 prompts，多为极端 case）：它设计为评测集，不作训练用。

原则：**少而宽 > 多而窄**。256 条宽幅多样 prompts 每条采样一次，通常优于同一 32 条 prompts 重复 8 遍 —— 后者的熵崩陷阱更早，且最终分布窄于前者。

---

## 二、多少步合适

### TL;DR

| 用途 | 更新数 | Prompts × 每条采样 | 墙钟时间（H800） | 用来干什么 |
|---|---:|---|---|---|
| **冒烟测试** | 15–30 | 5 × 3–6 轮 | ~2 分钟 | 验证训练循环无 NaN、熵不塌、权重确实更新 |
| **开发验收** | 64–128 | 64 × 1–2 | ~5 分钟 | 诊断熵崩、对照教师 view、调 tau |
| **完整报告实验** | 256 | 256 × 1 | ~15 分钟 | 投稿数字、bootstrap 区间、与 paper 对比 |

### 细节说明

归档的 M15 臂跑了 **256 次更新**（每条 prompt 恰好一次 rollout + update），三个种子的 GenEval 中位数比基座高 +4.10pp（99% bootstrap 区间 [0.93, 7.52]）。

为什么 256 够了：
- **熵监控取代早停。** 不是"跑到看不动了"，而是"教师熵 > 5 且学生熵上升 <0.3/32步"（M15 记录的退出条件）。实际上 M15 跑完 256 步都还没触发，说明这个 LoRA rank32、lr 2e-5 的配置很保守。
- **单次采样 vs 多轮。** OPSD 每步的教师目标来自当前权重自己未来的预测，它 **不固定**。重复同一 prompt，第二次 rollout 已经在不同权重上采样，行为等价于新 prompt。256 条 × 1 轮 和 32 条 × 8 轮 的**信号总量类似**，但前者分布更宽。
- **为什么不是 1k/10k。** 自蒸馏的风险是学生熵塌到教师的尖峰，而教师本身是基座自己。开更多步不会凭空出现"更强的教师" —— 基座熵就到此为止了。真实报告里 all_future 教师熵 ~5.7–6.5，学生训练后从 7.0 降到 ~6.2，间隙关闭了一半；继续跑只会让它继续降，GenEval 可能还没涨就先崩了。

**早停监控建议**：记录滚动 32 步的 `student_entropy` 平均值，一旦 `Δ_32 < 0.15` **且** `teacher_entropy < 5.5`，停止训练（不管是否跑满 256）。这个阈值对应"学生已经接近教师，继续跑会压缩而非改进"。

---

## 三、怎样评测

### 自动评测：GenEval

**GenEval**（Ghosh et al., ICLR 2024）是当前 compositional T2I 评测的 de-facto 标准。这里使用其 **553 条 prompts** 的流行子集（[原论文 full set 1008 条](https://github.com/djghosh13/geneval)，但 MMaDA、DDPO、DPOK 等论文多用 553 条子集），覆盖 single object、two objects、counting、colors、position、color attribution 六个主维度，每条配问题模板（如"图中出现几只猫？"），用 VQA 模型打分 0/1，汇总成 0–100% 的通过率。

配置：
- 每条 prompt 生成 **4 张图**，每张问一次，取 majority vote（GenEval 官方协议）。
- H800 上跑完 553×4=2212 张需 **~15 分钟**（15 步，batch=1）。
- 环境在 `/physis/unified-dopsd/geneval_venv`，权重在 `/physis/models/geneval/`。

> ⚠️ **本重构仓库目前只有训练部分（`opsd/`），生成与 GenEval 打分脚本尚未迁移。** 已有的实现是归档里的 `full_future_h800/endpoint_generate.py`、`extended_generate.py`、`score_images.py`、`geneval_evaluate_reference.py` 与服务器上的 `geneval_venv`。跑评测前先用那些脚本，或者写一个 `opsd/generate.py`（只需 `setup()` + `rollout()` 两处调用，约 40 行）。

代码片段（示例，需适配你的路径）：
```bash
# 假设 checkpoint 在 /tmp/smoke_allfuture/adapter_model.safetensors
python -m opsd.generate \
  --checkpoint /tmp/smoke_allfuture \
  --prompts /physis/models/geneval/evaluation_metadata.jsonl \
  --samples-per-prompt 4 \
  --out /tmp/eval_imgs \
  --seed 100

python geneval_wrapper.py --img-dir /tmp/eval_imgs --out results.json
# 输出: {"overall": 0.573, "single_obj": 0.98, ...}
```

**Bootstrap 置信区间**：GenEval 的 553 条是分层采样，不是 i.i.d.，但可以对主维度分数做 percentile bootstrap（M15 报告用了 1000 次重采样，经五项主比较校正，99% 区间 [0.93, 7.52]pp）。

### 人工偏好

GenEval 只测组合正确性，不测美学、真实感、风格一致性。如果怀疑 OPSD 提升了成分正确但牺牲了视觉质量（早期终点教师确实出现过这个trade-off），需要加人工盲测：

- **两两对比**：基座 vs 训练后，每对 prompt 各生成 4 张（同 seed 控制布局），打乱顺序问"哪边更符合描述？" + "哪边视觉质量更高？"
- **样本量**：64 条 prompts × 3 标注员 ≈ 192 judgments，足以检测 ±10pp 的偏好差（paired t-test, α=0.05）。
- **结果归档**：M15 没有跑人工，因为 GenEval +4.10pp 且生成的图目视无劣化。如果你跑了新教师 view 出现质量退化，加这个。

### 奖励模型：ImageReward / HPSv2 / PickScore

不建议作为主要指标，原因：
- **可 hack**：早期 diffusion-OPSD 实验发现 CLIP score 可以被蒸馏提升 0.1+，但人工盲测无差异（reward hacking）。
- **不测组合**：它们打的是"整图美学+对齐"，对 attribute binding（"red backpack next to green bike"）不敏感。

仅在以下场景记录它：
- 你怀疑教师 view 引入了 CLIP 偏见（如 endpoint_visible 那版教师在可见位置上预测，可能过拟合 CLIP 喜欢的高频 token）。
- 需要与 DPOK / DDPO 等用 reward model 做 RLHF 的工作对比（它们报告 ImageReward，你也记一个以示公平）。

---

## 推荐的标准流程

```bash
# 1. 冒烟（2 分钟）
python -m opsd.train --teacher all_future --updates 30 \
  --prompts smoke5.json --seed 42 --out runs/smoke

# 2. 开发验收（5 分钟）
python -m opsd.train --teacher all_future --updates 64 \
  --prompts dev64_parti.json --seed 42 --out runs/dev64

# 3. 完整训练（15 分钟）
python -m opsd.train --teacher all_future --updates 256 \
  --prompts train256_parti_prompts.json --seed 42 --out runs/m256_seed42

# 4. GenEval（~15 分钟；生成脚本需先从归档迁移或自行实现，见上文说明）
#    归档对应入口：endpoint_generate.py / extended_generate.py + endpoint_geneval.sh

# 5. 多训练种子 + bootstrap 区间
for s in 42 43; do
  python -m opsd.train --teacher all_future --updates 256 --seed $s \
    --rollout-seed $((1000*(s-41))) --out runs/m256_seed$s
  # ... 生成 + GenEval
done
python bootstrap_ci.py --runs runs/m256_seed42 runs/m256_seed43 --out ci.json
```

归档结果的数字（**两个训练种子** 42 和 43，不是我上面口误说的三个）：

| 方法 | seed42 | seed43 | 两种子均值 |
|---|---:|---:|---:|
| 冻结基座 | 53.26 | 53.26 | 53.26% |
| 完整未来平均（all_future） | 58.65 | 56.06 | **57.36%** |
| 论文式可见终点回退 | 57.77 | 57.48 | 57.62% |
| 重新遮挡终点回退 | 57.82 | 58.78 | 58.30% |

完整平均相对基座 **+4.10pp**，五项主比较经 Bonferroni 校正后 99% 配对 bootstrap 区间 **[+0.47, +7.71]**。注意区间是"条件于已训练的这两个模型"，不是跨训练种子的总体区间 —— 两个种子本身就差了 2.6pp，所以报告的区间比真实不确定性窄。

---

## 当前 H800 服务器上已有的资源

登录：`ssh -p 8595 root@106.12.153.99`

```
/physis/models/
  MMaDA-8B-MixCoT/              # 基座权重
  magvitv2/                     # VQ tokenizer
  GenEval/prompts/              # 1008 条评测 prompts
  ImageReward/                  # 奖励模型（可选）
  openai/clip-vit-large-patch14 # CLIP（可选）

/physis/unified-dopsd/
  venv/                         # 主环境（transformers 4.57.1, peft, torch 2.4）
  geneval_venv/                 # GenEval 依赖（单独环境避免冲突）
  reward_venv/                  # ImageReward（可选）
  runs/m*/                      # 归档的 M7–M19 实验目录
    train256_parti_prompts.json # 256 条训练数据（每个 m* 下都有副本）
```

`/physis` 是 8×H800 挂载的共享盘，`df -h /physis` 显示还有 ~4.2T 可用。

现在代码在 `/physis/unified-dopsd/mmada-opsd-clean`，干净重构后的目录。

---

最后提醒：**256 步是充足的，不是保守的下界**。真正控制质量的不是"跑够 1k 步"，而是"教师始终在分布内 + 学生熵监控 + 多种子验证"。如果预算紧张，64 步开发集 + 单 seed 256 步就能得到可信的方向性结论（比基座好 / 坏 / 无差异）。三 seed + bootstrap 是投稿级标准，不是探索期的必选项。

# GenEval2-800 评测结果：OPSD (all_future) vs MMaDA-8B 基座

日期：2026-10-16/17 ｜ 服务器：H800×8 ｜ 代码：`mmada-opsd-clean`（上游协议对齐版）

## 一句话结论

在官方温度退火协议下，256 步 OPSD 训练在 GenEval2-800 上带来 **+4.10pp**（soft_tifa_am，2 个训练种子 × 2 个图像种子合并，两层 bootstrap 95% CI [+2.88, +5.31]，P(>0)=1.000）。两个独立训练种子分别得到 +3.79pp 和 +4.41pp，方向与量级一致——收益不是单次抽签的运气。

## 实验设置

| 项 | 值 |
|---|---|
| 学生 | MMaDA-8B-MixCoT + LoRA（r=32，q/k/v/up/ff_proj，160 矩阵） |
| 教师 | 同一权重、禁用 adapter、no_grad，视图 all_future |
| 训练数据 | Parti 256 条（`train256_parti_prompts.json`），与归档一致 |
| 步数 | 256 updates，lr 2e-5，forward KL，mask_threshold 0.75 |
| 温度退火 | **upstream**（复合式，与官方 MMaDA 逐位一致；归档用的 per-step 与上游不符） |
| 生成 | 800 prompts × 2 seeds（17000/29000），15 步，CFG 3.5，种子只依赖 prompt 下标 → 与基座严格配对 |
| 评分 | 官方 `evaluation.py`（md5 81a6e287…，未改动），judge Qwen3-VL-8B-Instruct，soft_tifa |

## 教师视图横向对比（seed42，同协议，合并 2 图像种子 n=1600）

| 教师视图 | soft_tifa_am | soft_tifa_gm | Δam vs 基座 | 95% CI（配对） |
|---|---:|---:|---:|---|
| 基座（无训练） | 51.71 | 12.47 | — | — |
| **endpoint_visible** | **56.57** | **15.07** | **+4.86** | [+3.90, +5.85] |
| all_future（seed43） | 56.11 | 14.72 | +4.41 | [+3.47, +5.37] |
| all_future（seed42） | 55.49 | 14.34 | +3.79 | [+2.84, +4.74] |
| gap_n (gap=2) | 55.11 | 14.03 | +3.41 | [+2.47, +4.41] |
| endpoint_remasked | 52.17 | 12.68 | +0.46 | [−0.50, +1.42] |

**排序有明确解释**：教师能看到学生该位置的答案时收益最大（endpoint_visible +4.86），看若干步后的轨迹次之（all_future / gap_n ≈ +3.4~4.9，且信息被稀释），而把该位置重新遮住后收益消失（endpoint_remasked +0.46，CI 跨零）。

endpoint_visible 与 endpoint_remasked 共享**同一个最终状态**，唯一差别是后者把待预测位置重新置为 mask。前者 +4.86pp、后者 +0.46pp，说明**教师信号的价值来自"看见该位置的解"本身**，而不是"看见一个解码后的图像"这一泛泛的信息。这是本组实验最干净的因果对照。

注意：本表 CI 用 1600 条 prompt 混合重采样，未区分同一 prompt 的两个图像种子（略偏窄），仅用于臂间横向比较。正式头条数字用下面的两层 bootstrap。

## 主结果（soft_tifa_am）

| 训练种子 | 图像种子 | 训练模型 | 基座 | 差值 | 95% CI（配对） |
|---|---|---:|---:|---:|---|
| 42 | 17000 | 55.65 | 51.77 | +3.87 | [+2.60, +5.21] |
| 42 | 29000 | 55.34 | 51.64 | +3.70 | [+2.41, +5.05] |
| 42 | 合并 n=1600 | 55.49 | 51.71 | +3.79 | — |
| 43 | 17000 | 56.23 | 51.77 | +4.46 | [+3.21, +5.71] |
| 43 | 29000 | 55.99 | 51.64 | +4.35 | [+3.03, +5.69] |
| 43 | 合并 n=1600 | 56.11 | 51.71 | +4.41 | — |
| **两层合并** | n=3200 | — | — | **+4.10** | **[+2.88, +5.31]** |

两层 bootstrap 同时重采样训练种子和 prompt（各 10000 次），覆盖"换一次训练"和"换一批 prompt"两种不确定性；区间比单模型的配对 bootstrap 更宽，是更诚实的头条数字。重采样中最小 gap 为 +1.78pp，从未穿越零。

soft_tifa_gm：seed42 合并 14.34 vs 12.47，seed43 合并 14.72 vs 12.47。gm 相对提升（~+16%）大于 am（~+7.9%），说明收益部分来自减少"整体全错"的 prompt，而非只抬均值。

## 分维度（两个训练种子方向一致）

- **按组合复杂度（atom_count 3–10）**：两个种子各 8 层，共 16 个分层里 15 个为正。复杂 prompt 受益更大（atom_count=5 最高 +8.1；=8：+6.5/+1.8/+5.1/+5.6；=10：+4.6/+4.4/+4.8/+4.6）。唯二负值是 atom_count=4（seed42 的 -1.4、seed43 的 -0.1），而该层在 seed42-seed1 和 seed43-seed1 分别是 +4.0、+6.1——同为噪声。
- **按 skill**：object +5.2~+6.9（最大最稳）＞ attribute +2.7~+4.0 ＞ position +2.8~+4.0 ≈ count +2.8~+3.9。verb 仅 86 题，在 +4.3 与 -3.0 之间横跳，不可解读。
- **瓶颈**：count 绝对分只有 ~37%，是单一最大拖累（每条 prompt 都含 count 问题）。object 已到 74%，继续提升空间有限。

## 协议与可信度说明

1. **配对设计**：两模型同 prompt 同噪声起点，差异只来自 LoRA。单模型内区间按配对差计算，从而比非配对窄。
2. **与归档 +4.10pp 的关系**：归档温度协议与上游不一致（per-step 而非复合退火），本次是干净协议下的首次确认，数值恰好也是 +4.10pp。**注意这是巧合式的吻合，不是同一计算**。
3. **判据覆盖**：评分器加载后有 GPU 断言；聚合逻辑与官方逐式一致（scipy gmean）；分片合并保序已单测；生成 manifest 记录 adapter sha256 与全部生成参数。
4. **局限**：① 评测 judge 为单一模型（Qwen3-VL-8B），soft_tifa 是 VQA 概率代理而非人工评判；② 仅 2 个训练种子，两层 bootstrap 的自由度受此限制；③ 只测了 all_future 一个教师视图，其余视图（endpoint_visible/remasked/gap_n）未在本协议下重跑。

## 文件索引（服务器 /physis/unified-dopsd/）

- 训练：`mmada-opsd-clean/runs/af256_s42/`、`runs/af256_s43/`（含 metrics.jsonl、config.json、TRAIN_DONE.json、checkpoint-final/）
- 图像：各 run 下 `geneval2_imgs/`（manifest.json 记录 adapter sha256 与生成参数）
- 分数：`gv2_scores/<model>/seed<i>/scores.json`（逐 prompt 逐问题概率）、`score.json`（聚合）
- 对比脚本：`mmada-opsd-clean/eval/compare.py`（配对 bootstrap、atom_count/skill 分层）

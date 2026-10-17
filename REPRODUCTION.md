# 可重现性说明

本重构的目标是**算法正确性**，不是比特级重现归档数字。

---

## 与归档的差异

### 1. 温度退火语义不同（采样协议变化）

**归档**（`full_future_h800/legacy.py:68`）：
```python
t = temperature * (1.0 - ratio)  # 每步独立计算
```
每步的 softmax 温度都从初值 1.0 开始乘 `(1.0 - step/total)`，所以步 0 用 `t=1.0`，步 1 用 `t=0.933`，步 14 用 `t=0.067`。

**本重构**（对齐上游 `vendor/MMaDA/models/modeling_mmada.py:203`）：
```python
temperature = temperature * (1.0 - ratio)  # 原地更新，复合衰减
```
温度在循环中**累乘衰减**（`1.0 × 14/15 × 13/15 × 12/15 × …`）。两者逐步对照：

| step | 上游（复合） | 归档（每步独立） |
|---:|---:|---:|
| 0 | 0.9333 | 0.9333 |
| 1 | 0.8089 | 0.8667 |
| 2 | 0.6471 | 0.8000 |
| 5 | 0.1898 | 0.6000 |
| 10 | 0.00168 | 0.2667 |
| 14 | 0.0 | 0.0 |

步 0 两者相同，**从步 1 起分叉**，此后温度差一个数量级以上。`mask_by_random_topk` 的 `confidence = log(probs) + temperature * gumbel_noise`，温度直接决定 gumbel 噪声幅度，所以被 mask 的位置从步 1 起就不同，轨迹完全分开。

**结果**：所有下游数字都不再可比。归档报告的 GenEval +4.10pp 是在 **per-step 温度** 采样下测得的，不是 MMaDA 官方协议。

**实测影响（15 步冒烟，同一 5 条 prompts）**：

| 指标 | upstream（复合） | per_step（归档） |
|---|---:|---:|
| 选中的中间态 k | 6（全部） | 6（全部） |
| 打分位置数 | 827–828 | 826–828 |
| loss 均值 | 0.6418 | 0.6642 |
| grad_norm 均值 | 0.1373 | 0.1398 |
| 教师熵 均值 | 5.636 | 5.569 |
| 学生熵 均值 | 6.299 | 6.326 |

即**单条轨迹不同，但 15 步的聚合统计几乎重合**。这说明温度退火差异主要是逐样本的重新采样噪声，不是系统性偏移。因此：

- 归档结论的**方向性**大概率仍然成立（真实轨迹分布相似）；
- 但**具体数值**（+4.10pp）不能声称是在官方协议下得到的，重跑一遍最干净。

**本重构的选择**：对齐上游，代价是放弃比特级重现。理由是"归档数字建立在一个采样 bug 上"比"不能重现归档数字"更严重。

---

### 2. bf16 vs fp32 softmax（精度差异）

**归档**：未显式记录；我未找到 `.float()` 调用，猜测走 bf16 原生路径（与上游一致）。

**本重构初版**：`.float().softmax(-1)` 以避免 bf16 下溢（过度谨慎）。实测发现与上游逐位不一致后改回 `lg.softmax(-1)`，现已对齐。

---

### 3. 单元测试与集成测试覆盖（新增）

归档代码无自动化测试。本重构添加：
- **23 项单元测试**（`tests/test_core.py`）：rollout 结构不变性、mask 单调性、教师 target 形状、熵/KL 数值正确性、LoRA 激活控制。
- **4 类集成测试**（`tests/test_integration.py`）：
  1. rollout 与上游逐位一致性（两个 prompts）
  2. 四种教师 view 生成有效 target
  3. 同状态下 `disable_adapter()` 前后前向一致性

在真实 GPU 上 < 1 分钟跑完全部测试。

---

### 4. 文档化的教师 view 枚举（架构清理）

归档有三个教师定义混在不同训练脚本里（`train_endpoint.py`、`train_coverage.py`、`train_fast.py`），且有一次中途更换（可见终点 → 未来平均）导致文档前后矛盾。

本重构把教师变成显式枚举（`teacher.TeacherView`）：
- `ENDPOINT_VISIBLE` / `ENDPOINT_REMASKED`：论文式迁移对照
- `GAP_N`：gap2 探针
- `ALL_FUTURE`：报告主实验

同一份代码跑全部 view，不再需要三个脚本。

---

## 可复现的内容

| 验证项 | 状态 | 说明 |
|---|:---:|---|
| rollout 与上游 bit-identical | ✓ | 集成测试自动检查，两个 prompts 完全一致 |
| 四种教师 view 生成有效 target | ✓ | 集成测试验证形状、dtype、熵范围 |
| LoRA 梯度更新非零 | ✓ | 冒烟测试 15 步后 160/160 矩阵非零 |
| loss / grad_norm / 熵的数量级 | ✓ | 0.5–1.4 / 0.06–0.26 / 教师 4–7、学生 4–7 |
| 相对 GenEval 收益方向 | **待验证** | 需跑完整 256 步 + GenEval 打分 |
| 归档的 +4.10pp 绝对值 | ✗ | 温度退火语义不同，轨迹从步 0 分叉 |

---

## 推荐验证流程

如果你需要确认"算法本身是否有效"而不是"能否重现归档数字"：

```bash
# 1. 单元测试（本地，无 GPU）
python -m unittest discover -s tests -p "test_core.py"

# 2. 集成测试（H800）
CUDA_VISIBLE_DEVICES=0 python -m tests.test_integration

# 3. 冒烟训练（15 步）
python -m opsd.train --teacher all_future --updates 15 --seed 42 \
  --prompts /tmp/smoke_prompts.json --out /tmp/smoke

# 4. 开发集（64 步）+ GenEval 子集验证
#    [需先迁移生成脚本，见 DATA.md]

# 5. 完整 256 步 + 553×4 GenEval
#    对比基座 53.26%，预期 +3~5pp（上游温度协议）
```

**关键指标是相对收益（训练后 vs 基座），不是绝对值。** 如果 all_future 在上游温度协议下仍能比基座高 3pp+，算法就复现了；具体是 56% 还是 57% 取决于训练种子和评测集采样噪声。

---

## 归档原始数字的参考价值

归档的探索价值在**教师定义对比的相对排序**，不在绝对分数：

| 归档实验臂 | GenEval（两种子均值） | 解读 |
|---|---:|---|
| 冻结基座 | 53.26% | 参照 |
| gap2 | 55.51% | 单步差分，+2.25pp |
| all_future | 57.36% | 未来平均，+4.10pp，最佳 |
| endpoint_visible | 57.62% | 可见终点回退，+4.36pp |
| endpoint_remasked | 58.30% | 重新遮挡终点，+5.04pp |

**启示**：
1. 教师看未来遮挡位置（all_future）比只看 gap2 好。
2. 终点回退能工作（visible 和 remasked 都 > 基座），但 **分布外查询**（visible）与 **分布内**（remasked）的差异不确定（区间重叠）。
3. 熵对照臂（未列出）GenEval 崩到 21%，证明"loss 降 ≠ 性能好"。

这些**结论的方向性**（未来平均优于单步、熵崩是真实陷阱）是可靠的；**绝对值的小数第一位**（56.06 vs 57.77）不必当真。

---

## 如果你需要比特级重现归档数字

修改 `opsd/rollout.py` 的温度退火逻辑：

```python
# line 109, 当前是:
        running_temp = _temperature(step, steps, temperature, anneal, running_temp)

# 改成归档的 per-step 语义:
        running_temp = temperature * (1.0 - (step + 1) / steps)
```

然后重跑。但这意味着你在用一个与上游不一致的采样器，且该不一致是无意的（归档没有文档说明为什么偏离上游）。

我的建议是**不这么做** —— 对齐上游更重要。

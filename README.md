# MMaDA-OPSD

On-policy self-distillation (OPSD) for MMaDA-8B-MixCoT, where a model trained on discrete image tokens teaches itself by predicting future states during rollout and using those predictions as targets for earlier positions.

This codebase aligns the temperature schedule with MMaDA upstream (the archived experiments used a per-step anneal that diverged from the official implementation). Training for 256 updates on 256 Parti prompts (LoRA rank 32) yields **+4.10pp on GenEval2-800** (soft_tifa_am, two training seeds pooled, 95% CI [+2.88, +5.31]).

[Full experimental report (Feishu)](https://gcndfehc1xtb.feishu.cn/docx/T32hd45ltohSBYxnrJvcrWCWnpb) · [Archived experiments (GitHub)](https://github.com/lindaxhy/Unified-Discrete-MOPD)

---

## Results

GenEval2-800 prompts × 2 image seeds, official scorer with Qwen3-VL-8B judge:

| Teacher view | soft_tifa_am | vs base | 95% CI |
|---|---:|---:|---|
| base (frozen) | 51.71 | — | — |
| **endpoint_visible** | **56.57** | **+4.86** | [+3.90, +5.85] |
| all_future (s42/43 avg) | 55.80 | +4.10 | [+2.88, +5.31] |
| gap_n (gap=2) | 55.11 | +3.41 | [+2.47, +4.41] |
| endpoint_remasked | 52.17 | +0.46 | [−0.50, +1.42] |

**Key finding:** `endpoint_visible` and `endpoint_remasked` query the same final state, differing only in whether the target positions are re-masked before the teacher sees them. The former gains +4.86pp while the latter is statistically zero, isolating what makes the teacher signal useful: **seeing the answer at the position being scored**, not merely a decoded image.

---

## Installation

**Requirements:** Linux, CUDA GPU (experiments ran on H800 80GB), Python 3.10+.

1. Clone and install dependencies:

```bash
git clone https://github.com/lindaxhy/MMaDA-OPSD.git
cd MMaDA-OPSD
pip install torch torchvision peft transformers safetensors scipy pillow tqdm
```

2. Get the weights (not redistributed in this repo):

- **MMaDA-8B-MixCoT** — base model; see [MMaDA repo](https://github.com/Gen-Verse/MMaDA) for access
- **MAGVITv2** — image tokenizer; from the [MAGVIT repo](https://github.com/google-research/magvit)
- **Qwen3-VL-8B-Instruct** — GenEval2 judge; HuggingFace pulls it automatically if you have network, or download locally

3. Configure paths:

```bash
cp .env.example .env
# edit .env to point MMADA_OPSD_MODEL and MMADA_OPSD_VQ at your weights
set -a; source .env; set +a
```

The vendored `vendor/MMaDA` tree is the default source for model code and already matches the environment variable defaults; you only need to override if you have a different MMaDA checkout.

---

## Structure

```
opsd/           Core modules (train, rollout, teacher views, student, loss, MMaDA adapter)
eval/           GenEval2 scorer wrapper, bootstrap, comparison scripts
tests/          23 unit + integration tests
scripts/        Training, generation, and scoring pipelines
vendor/         MMaDA upstream source (33 files), geneval (8 files), geneval2 (scorer)
data/           geneval2_data.jsonl (800 prompts with compositional VQA questions)
```

Docs: `RESULTS.md` (detailed breakdown), `REPRODUCTION.md` (step-by-step replication), `DECISION.md` (design rationale in Chinese), `DATA.md` (datasets and splits).

---

## Quick start

### 1. Train (256 updates, ~15 min on H800)

```bash
python -m opsd.train --teacher all_future --prompts data/train256_parti_prompts.json \
  --updates 256 --seed 42 --out runs/af256_s42
```

Output: `runs/af256_s42/checkpoint-final/adapter_model.safetensors` (LoRA weights), `metrics.jsonl` (per-update loss / entropy).

### 2. Generate images

```bash
python -m opsd.generate --checkpoint runs/af256_s42/checkpoint-final \
  --prompts data/geneval2_data.jsonl --seeds 17000 29000 \
  --out runs/af256_s42/geneval2_imgs --steps 15 --cfg 3.5
```

Layout: `runs/af256_s42/geneval2_imgs/<prompt_idx:05d>/<seed_idx:05d>.png`.

### 3. Score

```bash
python -m eval.score_geneval2 \
  --benchmark data/geneval2_data.jsonl \
  --images runs/af256_s42/geneval2_imgs \
  --out runs/scores/af256_s42.json --method soft_tifa_am
```

Returns per-prompt VQA probabilities averaged over atoms (soft_tifa_am) or geometric mean (soft_tifa_gm).

### 4. Compare vs base

```bash
python -m eval.compare --root runs/scores --trained af256_s42 --base base \
  --data data/geneval2_data.jsonl
```

Prints paired bootstrap intervals and skill/complexity breakdowns.

---

## Teacher views

Implemented in `opsd/teacher.py`:

- **all_future** — average the teacher's predictions over all future unmasked states reachable from the current position (default, paper result +4.10pp)
- **endpoint_visible** — query the teacher at the final state with all positions visible (+4.86pp, best single-seed result)
- **endpoint_remasked** — query the final state but re-mask the target positions before the teacher sees them (+0.46pp, ablation control showing the answer visibility matters)
- **gap_n** — single-state teacher N steps ahead (gap=2 yields +3.41pp)

---

## Testing

```bash
python -m pytest tests/ -v
# or
python -m unittest discover -p 'test_*.py' -v
```

23 tests cover rollout correctness, teacher view arithmetic, student selection, loss computation, and end-to-end generation. Requires the base model weights; set `MMADA_OPSD_MODEL` or pass `--model` to `test_integration.py`.

---

## Citation

This code accompanies the GenEval2-800 re-run of discrete OPSD on MMaDA. The archived experiments (with protocol divergence) are at [`Unified-Discrete-MOPD`](https://github.com/lindaxhy/Unified-Discrete-MOPD). Upstream method references:

- **dOPSD** — Yin et al., "[On-Policy Self-Distillation for Improved Text-to-Image Generation](https://arxiv.org/abs/2607.04428)"
- **MMaDA** — [MMaDA GitHub](https://github.com/Gen-Verse/MMaDA)
- **GenEval2** — official scorer in `vendor/geneval2/evaluation.py`

---

## License

This project's code (opsd/, eval/, tests/, scripts/) is provided for research purposes. Third-party components retain their original licenses:

- `vendor/MMaDA` — see `vendor/MMaDA/LICENSE`
- `vendor/geneval` — see `vendor/geneval/LICENSE`
- `vendor/geneval2` — see `vendor/geneval2/LICENSE`

Model weights (MMaDA-8B-MixCoT, MAGVITv2, Qwen3-VL) are governed by their respective terms and are not included in this repository.

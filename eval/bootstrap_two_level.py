"""Two-level bootstrap over training seeds AND prompts.

The paired bootstrap in `compare.py` resamples prompts within one trained model, so
it answers "would this gap hold on other prompts?" It does not cover "would it hold
for another training run?" -- that is the level a paper reviewer will press on.

With two replicated training runs (seeds 42 and 43) we can resample both levels:
draw the two training seeds with replacement, and within each resample prompts with
replacement. That gives an interval reflecting both sources of variance, which is
necessarily wider and is the honest headline number.

    python -m eval.bootstrap_two_level --root runs/scores
"""
import argparse
import json
import random
from pathlib import Path

TRAIN_SEEDS = ["af256_s42", "af256_s43"]
IMAGE_SEEDS = [0, 1]
BASE = "base"


def load(root, model, image_seed):
    p = Path(root) / model / f"seed{image_seed}" / "scores.json"
    return json.loads(p.read_text()) if p.exists() else None


def prompt_mean(xs):
    return sum(xs) / len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/scores")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base_runs = [load(args.root, BASE, s) for s in IMAGE_SEEDS]
    trained = {}
    for m in TRAIN_SEEDS:
        runs = [load(args.root, m, s) for s in IMAGE_SEEDS]
        if all(r is not None for r in runs):
            trained[m] = runs

    if not trained:
        raise SystemExit("no trained runs found")
    if any(b is None for b in base_runs):
        raise SystemExit("base runs incomplete")

    base_pool = [prompt_mean(s) for run in base_runs for s in run]
    base_mean = sum(base_pool) / len(base_pool)

    # point estimates per training seed
    print("per training seed (pooled over 2 image seeds, n=1600 each):")
    per_seed = {}
    for m, runs in trained.items():
        t = [prompt_mean(s) for run in runs for s in run]
        m_t = sum(t) / len(t)
        per_seed[m] = (t, m_t)
        print(f"  {m}: trained {100*m_t:.2f}  base {100*base_mean:.2f}  "
              f"diff {100*(m_t-base_mean):+.2f}pp")

    if len(per_seed) < 2:
        print("\nonly one training seed available -- two-level bootstrap needs >= 2")
        return

    rng = random.Random(args.seed)
    names = sorted(per_seed)
    gaps = []
    for _ in range(args.n):
        picks = [rng.choice(names) for _ in names]
        seed_gaps = []
        for m in picks:
            t = per_seed[m][0]
            idx = [rng.randrange(len(t)) for _ in range(len(t))]
            t_res = sum(t[i] for i in idx) / len(idx)
            idx_b = [rng.randrange(len(base_pool)) for _ in range(len(base_pool))]
            b_res = sum(base_pool[i] for i in idx_b) / len(idx_b)
            seed_gaps.append(t_res - b_res)
        gaps.append(sum(seed_gaps) / len(seed_gaps))
    gaps.sort()

    mean_gap = sum(per_seed[m][1] for m in names) / len(names) - base_mean
    lo, hi = gaps[int(0.025 * args.n)], gaps[int(0.975 * args.n)]
    print(f"\n=== two-level bootstrap ({len(names)} training seeds x prompts, n={args.n}) ===")
    print(f"  mean gap   {100*mean_gap:+.2f}pp")
    print(f"  95% CI     [{100*lo:+.2f}, {100*hi:+.2f}]")
    print(f"  P(> 0)     {sum(1 for g in gaps if g > 0)/len(gaps):.3f}")
    print(f"  gap range across resamples: {100*gaps[0]:+.2f} .. {100*gaps[-1]:+.2f}")


if __name__ == "__main__":
    main()

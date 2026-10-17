"""Cross-arm summary: every trained variant against the shared base, one table.

Each arm shares the base generations and generation settings, so all arms are
directly comparable. Reports pooled soft_tifa_am/gm over both image seeds plus a
paired bootstrap against base for each.

    python -m eval.summarize_arms --root runs/scores
"""
import argparse
import json
import random
from pathlib import Path

ARMS = [
    ("af256_s42", "all_future"),
    ("ev256_s42", "endpoint_visible"),
    ("er256_s42", "endpoint_remasked"),
    ("g2256_s42", "gap_n (gap=2)"),
    ("af256_s43", "all_future (seed 43)"),
]
IMAGE_SEEDS = (0, 1)
BASE = "base"


def load(root, model, seed):
    p = Path(root) / model / f"seed{seed}" / "scores.json"
    return json.loads(p.read_text()) if p.exists() else None


def prompt_mean(xs):
    return sum(xs) / len(xs)


def gmean(xs):
    import math
    return math.exp(sum(math.log(max(v, 1e-12)) for v in xs) / len(xs))


def agg(runs, method):
    flat = [prompt_mean(s) if method == "am" else gmean(s) for run in runs for s in run]
    return 100 * sum(flat) / len(flat)


def paired(run_t, run_b, n=10000, seed=0):
    a = [prompt_mean(s) for run in run_t for s in run]
    b = [prompt_mean(s) for run in run_b for s in run]
    rng = random.Random(seed)
    gaps = []
    for _ in range(n):
        idx = [rng.randrange(len(a)) for _ in range(len(a))]
        gaps.append(sum(a[i] - b[i] for i in idx) / len(idx))
    gaps.sort()
    return (100 * sum(x - y for x, y in zip(a, b)) / len(a),
            gaps[int(0.025 * n)] * 100, gaps[int(0.975 * n)] * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/scores")
    ap.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()

    base_runs = [load(args.root, BASE, s) for s in IMAGE_SEEDS]
    if any(r is None for r in base_runs):
        raise SystemExit("base scores incomplete")

    print(f"{'arm':24s} {'am':>7s} {'gm':>7s} {'Δam':>7s} {'95% CI':>18s}")
    print(f"{'base':24s} {agg(base_runs,'am'):7.2f} {agg(base_runs,'gm'):7.2f}")
    print("-" * 68)
    for key, label in ARMS:
        runs = [load(args.root, key, s) for s in IMAGE_SEEDS]
        if any(r is None for r in runs):
            print(f"{label:24s} {'--':>7s} {'--':>7s}   (not scored yet)")
            continue
        am, gm = agg(runs, "am"), agg(runs, "gm")
        d, lo, hi = paired(runs, base_runs, n=args.n)
        print(f"{label:24s} {am:7.2f} {gm:7.2f} {d:+7.2f}   [{lo:+5.2f}, {hi:+5.2f}]")


if __name__ == "__main__":
    main()

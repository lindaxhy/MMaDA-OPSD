"""Compare GenEval2 scores between the base and OPSD-trained checkpoints.

Because `opsd.generate` derives its seed only from the prompt index, the two
models were sampled with identical noise per prompt -- the runs are paired. So
alongside the headline means this reports a paired bootstrap over prompts, which
is much tighter than an unpaired interval and is the honest way to say whether
the gap survives resampling.

    python -m eval.compare --root runs/scores \
        --data data/geneval2_data.jsonl
"""
import argparse
import json
import math
import random
from pathlib import Path

def load_scores(root, model, seed):
    p = Path(root) / model / f"seed{seed}" / "scores.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def prompt_mean(per_question):
    return sum(per_question) / len(per_question)


def gmean(xs):
    return math.exp(sum(math.log(max(v, 1e-12)) for v in xs) / len(xs))


def aggregate(lists, method):
    if method == "gm":
        per = [gmean(s) for s in lists]
    else:
        per = [prompt_mean(s) for s in lists]
    return 100 * sum(per) / len(per)


def paired_bootstrap(a, b, n=10000, seed=0):
    """Fraction of resamples where a beats b, plus the mean gap and 95% CI.

    a and b are per-prompt scores in the same prompt order (paired design).
    """
    assert len(a) == len(b), f"unpaired lengths: {len(a)} vs {len(b)}"
    rng = random.Random(seed)
    n_p = len(a)
    gaps = []
    for _ in range(n):
        idx = [rng.randrange(n_p) for _ in range(n_p)]
        gaps.append(sum(a[i] - b[i] for i in idx) / n_p)
    gaps.sort()
    mean_gap = sum(a[i] - b[i] for i in range(n_p)) / n_p
    lo, hi = gaps[int(0.025 * n)], gaps[int(0.975 * n)]
    wins = sum(1 for g in gaps if g > 0) / n
    return mean_gap, lo, hi, wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/scores")
    ap.add_argument("--data", default="data/geneval2_data.jsonl")
    ap.add_argument("--trained", default="af256_s42")
    ap.add_argument("--base", default="base")
    args = ap.parse_args()

    bench = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]

    for seed in (0, 1):
        t = load_scores(args.root, args.trained, seed)
        b = load_scores(args.root, args.base, seed)
        if t is None or b is None:
            print(f"seed{seed}: missing ({'trained ok' if t else 'trained MISSING'}, "
                  f"{'base ok' if b else 'base MISSING'})")
            continue
        print(f"=== seed{seed} (n={len(t)} prompts) ===")
        for meth in ("am", "gm"):
            at, ab = aggregate(t, meth), aggregate(b, meth)
            print(f"  {meth}: trained {at:.2f}   base {ab:.2f}   diff {at-ab:+.2f}")
        # paired bootstrap on the arithmetic mean, per prompt
        pa = [prompt_mean(s) for s in t]
        pb = [prompt_mean(s) for s in b]
        gap, lo, hi, wins = paired_bootstrap(pa, pb)
        print(f"  paired bootstrap (am): gap {100*gap:+.2f}pp  "
              f"95% CI [{100*lo:+.2f}, {100*hi:+.2f}]  P(win)={wins:.3f}")
        # per atom-count strata
        print("  by atom_count:")
        strata = {}
        for i, d in enumerate(bench):
            strata.setdefault(d["atom_count"], []).append(i)
        for k in sorted(strata):
            idx = strata[k]
            at = 100 * sum(pa[i] for i in idx) / len(idx)
            ab = 100 * sum(pb[i] for i in idx) / len(idx)
            print(f"    atom_count={k:2d} (n={len(idx):3d}): "
                  f"trained {at:.1f}  base {ab:.1f}  diff {at-ab:+.1f}")

        # per skill: each vqa question carries a skill label parallel to it
        print("  by skill:")
        buckets = {}
        for i, d in enumerate(bench):
            skills = d.get("skills") or []
            for j, sk in enumerate(skills):
                if j < len(t[i]) and j < len(b[i]):
                    buckets.setdefault(sk, []).append((t[i][j], b[i][j]))
        for sk in sorted(buckets, key=lambda s: -len(buckets[s])):
            pairs = buckets[sk]
            at = 100 * sum(x for x, _ in pairs) / len(pairs)
            ab = 100 * sum(y for _, y in pairs) / len(pairs)
            print(f"    {sk:10s} (n={len(pairs):5d}): "
                  f"trained {at:.1f}  base {ab:.1f}  diff {at-ab:+.1f}")

    # pooled
    ts = [load_scores(args.root, args.trained, s) for s in (0, 1)]
    bs = [load_scores(args.root, args.base, s) for s in (0, 1)]
    if all(x is not None for x in ts + bs):
        flat_t = [prompt_mean(s) for run in ts for s in run]
        flat_b = [prompt_mean(s) for run in bs for s in run]
        print(f"\n=== pooled over both seeds (n={len(flat_t)}) ===")
        for meth in ("am", "gm"):
            print(f"  {meth}: trained {aggregate([s for r in ts for s in r], meth):.2f}   "
                  f"base {aggregate([s for r in bs for s in r], meth):.2f}")


if __name__ == "__main__":
    main()

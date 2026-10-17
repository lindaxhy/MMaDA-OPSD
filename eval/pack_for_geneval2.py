"""Flatten generated images into the {prompt: filepath} dict the official GenEval2
scorer expects.

The official `evaluation.py` takes one image per prompt. We generate several seeds
per prompt, so this writes one dict per seed index and a matching benchmark jsonl
that duplicates each entry with a seed-suffixed prompt key -- keeping the official
scorer unmodified while letting us score every seed.

    python eval/pack_for_geneval2.py --images DIR --data geneval2_data.jsonl \
        --out-dir /tmp/gv2 --num-seeds 2

Writes /tmp/gv2/seed{0,1}/image_paths.json and /tmp/gv2/seed{0,1}/benchmark.jsonl.
"""
import argparse
import json
from pathlib import Path

SEED_KEY = "___seed"  # separator unlikely to appear in prompt text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True,
                    help="root written by opsd.generate (<root>/<idx:05d>/<seed:05d>.png)")
    ap.add_argument("--data", required=True, help="geneval2_data.jsonl (800 entries)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-seeds", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.images)
    data = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    if args.limit:
        data = data[:args.limit]

    out_root = Path(args.out_dir)
    for seed_idx in range(args.num_seeds):
        image_map = {}
        bench = []
        missing = 0
        for i, entry in enumerate(data):
            prompt = entry["prompt"]
            img = root / f"{i:05d}" / f"{seed_idx:05d}.png"
            if not img.exists():
                missing += 1
                continue
            key = f"{prompt}{SEED_KEY}{seed_idx}"
            image_map[key] = str(img.resolve())
            bench.append({**entry, "prompt": key})
        d = out_root / f"seed{seed_idx}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "image_paths.json").write_text(json.dumps(image_map))
        with (d / "benchmark.jsonl").open("w") as f:
            for b in bench:
                f.write(json.dumps(b) + "\n")
        print(f"seed{seed_idx}: {len(image_map)} images mapped, {missing} missing")
        if missing:
            print(f"  WARNING: {missing} prompts had no image for seed {seed_idx}")

    print(f"\nwrote under {out_root}. Score each seed with the official script:")
    print(f"  python evaluation.py --benchmark_data {out_root}/seed0/benchmark.jsonl \\")
    print(f"    --image_filepath_data {out_root}/seed0/image_paths.json \\")
    print(f"    --method soft_tifa_am --output_file {out_root}/seed0/scores.json")


if __name__ == "__main__":
    main()

"""Shardable driver around the official GenEval2 scorer.

`evaluation.py` is the upstream scorer (Meta) and is used unmodified in spirit:
we import its `soft_tifa` / `tifa` / `vqa_score` and its prompt construction, so
the metric is exactly the published one. Two things it cannot do that we need:

  * point at a local weights directory (it hardcodes the HF repo id, and its
    model loads at import time), and
  * split 800 prompts across processes (each prompt costs ~7.5 single-token VQA
    forwards, so one GPU is the bottleneck).

So we monkeypatch the module's globals before calling into it, and shard
contiguous blocks of the benchmark, merging the per-prompt score lists after.

Usage (per seed, per model):
    python -m eval.score_geneval2 --benchmark bench.jsonl --images image_paths.json \
        --out scores.json --method soft_tifa_am --shard-index 0 --num-shards 4
Then merge:
    python -m eval.score_geneval2 --merge --out-dir /tmp/gv2/seed0 --shards 4 \
        --method soft_tifa_am

Paths default to the layout this repo expects (vendor/geneval2, data/), and can be
overridden with MMADA_OPSD_GENEVAL2_SRC and MMADA_OPSD_QWEN.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from scipy.stats import gmean

_HERE = Path(__file__).resolve().parent.parent
GENEVAL2_SRC = Path(os.environ.get("MMADA_OPSD_GENEVAL2_SRC", _HERE / "vendor" / "geneval2"))
QWEN_DEFAULT = os.environ.get("MMADA_OPSD_QWEN", "Qwen/Qwen3-VL-8B-Instruct")


def load_scorer(model_dir):
    """Import upstream `evaluation`, redirecting its repo-id load to a local dir.

    The upstream module builds processor and model at import time from the literal
    id "Qwen/Qwen3-VL-8B-Instruct", with `device_map="auto"`. We rewrite that id to
    local weights and pin the device to whatever CUDA_VISIBLE_DEVICES already
    selects, so one shard occupies exactly one GPU. Everything else about the
    scoring passes through untouched, keeping the reference implementation
    byte-identical on disk while letting an offline machine run it.
    """
    if str(GENEVAL2_SRC) not in sys.path:
        sys.path.insert(0, str(GENEVAL2_SRC))
    import torch
    import transformers as tf

    repo_id = "Qwen/Qwen3-VL-8B-Instruct"
    orig_proc, orig_model = tf.AutoProcessor.from_pretrained, \
        tf.Qwen3VLForConditionalGeneration.from_pretrained

    def proc(ident, *a, **k):
        k["device_map"] = None
        k.pop("torch_dtype", None)
        return orig_proc(model_dir if ident == repo_id else ident, *a, **k)

    def model(ident, *a, **k):
        k.pop("device_map", None)
        return orig_model(model_dir if ident == repo_id else ident, *a,
                          device_map=None, **k).to("cuda")

    tf.AutoProcessor.from_pretrained = staticmethod(proc)
    tf.Qwen3VLForConditionalGeneration.from_pretrained = staticmethod(model)
    try:
        import evaluation
    finally:
        tf.AutoProcessor.from_pretrained = orig_proc
        tf.Qwen3VLForConditionalGeneration.from_pretrained = orig_model
    assert str(evaluation.qwen_model.device).startswith("cuda"), \
        f"scorer not on GPU: {evaluation.qwen_model.device}"
    return evaluation


def score_block(ev, entries, image_data, method):
    """Run the official per-prompt scorer over a list of benchmark entries."""
    score_fn = {"vqascore": lambda d, p: [ev.vqa_score(d["prompt"], p)],
                "tifa": lambda d, p: ev.tifa(d["vqa_list"], p)[1],
                "soft_tifa_am": lambda d, p: ev.soft_tifa(d["vqa_list"], p)[1],
                "soft_tifa_gm": lambda d, p: ev.soft_tifa(d["vqa_list"], p)[1]}[method]
    out = []
    for d in entries:
        if d["prompt"] not in image_data:
            raise SystemExit(f"missing image for prompt: {d['prompt']}")
        out.append(score_fn(d, image_data[d["prompt"]]))
    return out


def aggregate(all_score_lists, method):
    if method == "soft_tifa_gm":
        per_prompt = [gmean(s) for s in all_score_lists]
    else:
        per_prompt = [sum(s) / len(s) for s in all_score_lists]
    return 100 * sum(per_prompt) / len(per_prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", help="benchmark jsonl (official format)")
    ap.add_argument("--images", help="image_paths.json: {prompt: path}")
    ap.add_argument("--out", help="write per-prompt score lists here (this shard)")
    ap.add_argument("--model", default=QWEN_DEFAULT)
    ap.add_argument("--method", default="soft_tifa_am",
                    choices=["vqascore", "tifa", "soft_tifa_am", "soft_tifa_gm"])
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true",
                    help="merge shard outputs instead of scoring")
    ap.add_argument("--out-dir", help="merge mode: dir holding shard*.json")
    ap.add_argument("--shards", type=int, help="merge mode: number of shards")
    args = ap.parse_args()

    if args.merge:
        lists = []
        for s in range(args.shards):
            lists.extend(json.loads((Path(args.out_dir) / f"shard{s}.json").read_text()))
        total = aggregate(lists, args.method)
        (Path(args.out_dir) / "scores.json").write_text(json.dumps(lists))
        (Path(args.out_dir) / "score.json").write_text(
            json.dumps({"method": args.method, "score": total,
                        "n_prompts": len(lists)}, indent=2))
        print(f"{args.method}: {total:.4f}  (n={len(lists)})")
        return

    entries = [json.loads(l) for l in Path(args.benchmark).read_text().splitlines() if l.strip()]
    image_data = json.loads(Path(args.images).read_text())
    n = len(entries)
    start = (n * args.shard_index) // args.num_shards
    end = (n * (args.shard_index + 1)) // args.num_shards
    block = entries[start:end]

    ev = load_scorer(args.model)
    lists = score_block(ev, block, image_data, args.method)
    Path(args.out).write_text(json.dumps(lists))
    print(f"shard {args.shard_index}/{args.num_shards}: "
          f"{len(lists)} prompts ({start}..{end-1}) -> {args.out}")


if __name__ == "__main__":
    main()

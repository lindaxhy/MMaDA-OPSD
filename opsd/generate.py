"""Batch image generation from a trained checkpoint, and the evaluation glue.

Training (`opsd.train`) writes a LoRA checkpoint; this module turns it into PNGs
for a prompt set. Decoding uses the same MAGVITv2 path as training's `mmada_io`,
so generated images are exactly what the trained model produces.

Evaluation itself lives in `eval/` (GenEval2 protocol, official scorer); this
module only produces the images and a manifest that records exactly how.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from .mmada_io import load_for_generation
from .rollout import rollout


def decode_codes(vq, codes):
    """Decode (B, N_VQ) codebook ids to PIL images, matching training's decode."""
    x = vq.decode_code(codes.clamp(0, 8191)).detach()
    x = ((x.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1).cpu().numpy().astype("uint8")
    return [Image.fromarray(arr) for arr in x]


def load_prompts(path):
    """Read prompts from JSON (a list) or JSONL (one object per line).

    Accepts plain strings, objects with a 'prompt' key, or — for GenEval2-style
    files — objects with 'vqa_list'/'skills' alongside 'prompt'. Raises rather
    than guessing if it cannot find a prompt field, so a wrong dataset fails loudly.
    """
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list) or not data:
        raise ValueError(f"no prompts found in {path}")
    out = []
    for entry in data:
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, dict) and "prompt" in entry:
            out.append(entry["prompt"])
        else:
            raise ValueError(f"unrecognized prompt entry in {path}: {type(entry)} "
                             f"{list(entry)[:6] if isinstance(entry, dict) else entry}")
    return out


@torch.no_grad()
def generate_images(model, uni, vq, prompts, seeds, out_dir,
                    cfg=3.5, steps=15, seed_offset=0, resume=True,
                    shard_index=0, num_shards=1, index_offset=0):
    """Write PNGs as <out_dir>/<global_idx:05d>/<seed_idx:05d>.png.

    Layout mirrors the archived generator so existing scorers and manifests keep
    working: one directory per prompt, one file per seed. `shard_index`/`num_shards`
    split the prompt list across processes while keeping global prompt indices
    (via contiguous blocks), so sharded runs write into one shared tree and the
    evaluator sees a complete index space regardless of how generation was split.

    Returns the number of images written this call.
    """
    out = Path(out_dir)
    n = len(prompts)
    start = (n * shard_index) // num_shards
    end = (n * (shard_index + 1)) // num_shards
    written = 0
    for i in tqdm(range(start, end), desc=f"gen[{shard_index}/{num_shards}]"):
        prompt = prompts[i]
        gidx = index_offset + i
        dest = out / f"{gidx:05d}"
        dest.mkdir(parents=True, exist_ok=True)
        meta = dest / "metadata.jsonl"
        if not meta.exists():
            meta.write_text(json.dumps({"prompt": prompt, "index": gidx}) + "\n")
        for j, base_seed in enumerate(seeds):
            path = dest / f"{j:05d}.png"
            if resume and path.exists():
                continue
            seed = base_seed + seed_offset + gidx
            final, _hist, _template = rollout(
                model, uni, prompt, model.config.mask_token_id,
                steps=steps, cfg=cfg, seed=seed)
            img = decode_codes(vq, final.reshape(1, -1))[0]
            tmp = path.with_suffix(".tmp.png")
            img.save(tmp)
            tmp.replace(path)
            written += 1
    return written


def main():
    p = argparse.ArgumentParser(description="Generate images from a trained checkpoint")
    p.add_argument("--checkpoint", required=True,
                   help="checkpoint dir (contains adapter_model.safetensors), or 'base'")
    p.add_argument("--prompts", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[17000, 29000])
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--cfg", type=float, default=3.5)
    p.add_argument("--limit", type=int, default=0, help="only first N prompts (smoke)")
    p.add_argument("--shard-index", type=int, default=0,
                   help="this process's shard (0..num_shards-1)")
    p.add_argument("--num-shards", type=int, default=1,
                   help="split prompts across this many processes; all write to --out")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    torch.set_num_threads(4)
    use_adapter = args.checkpoint != "base"
    if use_adapter:
        ckpt = Path(args.checkpoint)
        if not (ckpt / "adapter_model.safetensors").exists():
            raise SystemExit(f"no adapter_model.safetensors in {ckpt}")

    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[:args.limit]

    model, uni, vq = load_for_generation(args.checkpoint, device=args.device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = generate_images(model, uni, vq, prompts, args.seeds, out,
                              cfg=args.cfg, steps=args.steps,
                              seed_offset=args.seed_offset,
                              shard_index=args.shard_index,
                              num_shards=args.num_shards)

    adapter = Path(args.checkpoint) / "adapter_model.safetensors"
    manifest = {
        "checkpoint": str(args.checkpoint),
        "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest()
                          if adapter.exists() else None,
        "prompts_file": args.prompts,
        "prompts_sha256": hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
        "n_prompts": len(prompts),
        "seeds": args.seeds,
        "seed_offset": args.seed_offset,
        "steps": args.steps,
        "cfg": args.cfg,
        "written_this_call": written,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    expected = len(prompts) * len(args.seeds)
    actual = len(list(out.glob("*/[0-9]*.png")))
    print(f"{actual}/{expected} images present in {out}")
    if actual == expected:
        (out / "DONE").write_text(f"{actual}\n")


if __name__ == "__main__":
    main()

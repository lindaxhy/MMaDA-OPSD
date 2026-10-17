"""Unified OPSD training loop for MMaDA image generation.

One update:
  rollout (on-policy, current LoRA) -> pick intermediate state k -> build detached
  teacher target -> forward the student at hist[k] -> KL on the scored positions
  -> LoRA update.

Determinism: `--deterministic` turns on torch.use_deterministic_algorithms and
disables the cudnn benchmark paths, which the reported runs used. Without it,
rollouts are not bit-reproducible across GPU occupancy. The rollout RNG and the
student-selection RNG are separate streams (`--rollout-seed`, `--selection-seed`)
so changing selection mode does not perturb the trajectories.

Every run refuses to write into a directory that already has a metrics.jsonl.
"""
import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

if "--deterministic" in __import__("sys").argv:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from .loss import entropy, forward_kl, one_sided, reverse_kl  # noqa: E402
from .mmada_io import N_VQ, logits_at, probs_at, setup  # noqa: E402
from .rollout import check_monotonic, mask_fractions, rollout  # noqa: E402
from .student import DEFAULT_THRESHOLD, choose_student  # noqa: E402
from .teacher import TeacherView, build_target  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MMaDA on-policy self-distillation")
    p.add_argument("--teacher", required=True, choices=[v.value for v in TeacherView])
    p.add_argument("--endpoint-mode", default="visible", choices=["visible", "remasked"],
                   help="only used by all_future fallback and the endpoint arms")
    p.add_argument("--gap", type=int, default=2, help="only used by --teacher gap_n")
    p.add_argument("--student-selection", default="fixed",
                   choices=["fixed", "least", "most", "uniform"])
    p.add_argument("--mask-threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--loss", default="forward_kl",
                   choices=["forward_kl", "reverse_kl", "one_sided"])
    p.add_argument("--updates", type=int, default=256)
    p.add_argument("--steps", type=int, default=15, help="diffusion decoding steps")
    p.add_argument("--cfg", type=float, default=3.5)
    p.add_argument("--anneal", default="upstream", choices=["upstream", "per_step"],
                   help="rollout temperature schedule; 'upstream' compounds as MMaDA's "
                        "t2i_generate does, 'per_step' is the pre-rewrite behaviour")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rollout-seed", type=int, default=1000)
    p.add_argument("--selection-seed", type=int, default=300000)
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_cudnn_sdp(False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.jsonl").exists():
        raise RuntimeError(f"refuse to mix an existing run: {out/'metrics.jsonl'}")

    prompts = json.loads(Path(args.prompts).read_text())
    if not prompts:
        raise ValueError("prompt file is empty")
    view = TeacherView(args.teacher)

    config = {
        **vars(args),
        "n_prompts": len(prompts),
        "lora_rank": 32,
        "student": "MMaDA-8B-MixCoT + LoRA",
        "teacher": "same weights, adapter disabled, no_grad",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    model, uni, mask_id = setup(args.seed, training=True)
    offset = len(uni.text_tokenizer)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    for it in range(args.updates):
        started = time.time()
        model.eval()
        prompt = prompts[it % len(prompts)]
        final, hist, template = rollout(
            model, uni, prompt, mask_id, steps=args.steps, cfg=args.cfg,
            seed=args.rollout_seed + it, anneal=args.anneal)
        check_monotonic(hist)

        fracs = mask_fractions(hist)
        k = choose_student(fracs, args.student_selection, args.mask_threshold,
                           args.selection_seed + it)
        early = hist[k]["masked"][0]

        q, counts, scored, info = build_target(
            model, template, hist, k, offset, mask_id, view, probs_at, final,
            gap=args.gap, endpoint_mode=args.endpoint_mode)
        assert q.requires_grad is False and scored.any(), "no scored positions"

        model.train()
        student_logits = logits_at(model, template, hist[k]["state"][0], offset, mask_id)
        same_state_kl = None
        if it == 0:
            # sanity: the no-grad and grad forwards must agree on the same state.
            # Compare logits to logits -- comparing a logit tensor against a
            # probability tensor through KL is meaningless and produces a huge number.
            model.eval()
            with torch.no_grad():
                check_logits = logits_at(model, template, hist[k]["state"][0], offset, mask_id)
            same_state_kl = float(
                (student_logits[early].float() - check_logits[early].float()).abs().max())
            assert same_state_kl < 1e-4, f"forward pass mismatch: {same_state_kl}"
            model.train()
            del check_logits

        if args.loss == "forward_kl":
            per_pos = forward_kl(q[scored], student_logits[scored])
            loss = per_pos.mean()
        elif args.loss == "reverse_kl":
            per_pos = reverse_kl(q[scored], student_logits[scored])
            loss = per_pos.mean()
        else:
            sampled_tokens = hist[k]["state"][0][scored].clamp_min(0)
            per_pos = one_sided(q[scored], student_logits[scored], sampled_tokens)
            loss = per_pos.mean()
        assert torch.isfinite(loss), f"non-finite loss at iter {it}"

        opt.zero_grad(set_to_none=True)
        loss.backward()
        assert all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n), \
            "a non-LoRA parameter received a gradient"
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        assert torch.isfinite(grad_norm), f"non-finite grad norm at iter {it}"
        opt.step()

        if it % args.log_every == 0 or it == args.updates - 1:
            with torch.no_grad():
                row = {
                    "iter": it,
                    "prompt": prompt,
                    "chosen": k,
                    "n_early": int(early.sum()),
                    "n_scored": int(scored.sum()),
                    "loss": float(loss),
                    "grad_norm": float(grad_norm),
                    "teacher_entropy": entropy(q[scored]),
                    "student_entropy": entropy(student_logits[scored].softmax(-1)),
                    "future_count_mean": info.get("future_count_mean"),
                    "future_count_hist": torch.bincount(counts[scored]).tolist(),
                    "teacher_forwards": info["teacher_forwards"],
                    "same_state_kl_max": same_state_kl,
                    "seconds": time.time() - started,
                    "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                    "info": {k2: v for k2, v in info.items() if k2 != "view"},
                }
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        del student_logits, q, loss, hist

    model.save_pretrained(out / "checkpoint-final")
    lora_b = [p for n, p in model.named_parameters() if "lora_B" in n]
    status = {
        "updates": args.updates,
        "lora_B_count": len(lora_b),
        "nonzero": sum(bool(x.detach().abs().sum() > 0) for x in lora_b),
        "prompts": len(prompts),
        "teacher": args.teacher,
    }
    assert status["nonzero"] == status["lora_B_count"], "some LoRA matrices never updated"
    (out / "TRAIN_DONE.json").write_text(json.dumps(status))
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()

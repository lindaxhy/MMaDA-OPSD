"""Integration checks that need the real MMaDA weights on a GPU.

    python -m tests.test_integration --model $MMaDA-OPSD_MODEL

Three things are verified, in increasing strength:

 1. rollout structure     -- hist has `steps` entries, hist[0] is all-masked, the
                            masked set shrinks monotonically, final is fully decoded
                            and all ids are in 0..8191.
 2. upstream equivalence   -- given the same prompt and the same generator seed,
                            our rollout's final ids equal the ids returned by
                            upstream `MMadaModelLM.t2i_generate` exactly. This is
                            the check that proves the mirrored sampler is faithful;
                            without it, every later number is suspect.
 3. teacher target shape  -- for each TeacherView, the target is a normalized
                            detached distribution, scored positions are a subset of
                            the student's masked positions, and every scored position
                            has either a future contribution or the fallback.

Run 1 and 3 always. Run 2 unless `--skip-upstream` is passed.
"""
import argparse
import sys

import torch

from opsd.mmada_io import N_VQ, MODEL_PATH, logits_at, probs_at, setup
from opsd.rollout import check_monotonic, mask_fractions, rollout
from opsd.teacher import TeacherView, build_target, surviving_mask

PROMPTS = [
    "a volcano erupting at night",
    "a green backpack and a pig",
]


def check_structure(final, hist, mask_id, steps):
    assert len(hist) == steps, f"expected {steps} recorded states, got {len(hist)}"
    assert bool(hist[0]["masked"].all()), "hist[0] must be fully masked"
    assert not bool((final == mask_id).any()), "final state still has mask ids"
    assert bool(((final >= 0) & (final < 8192)).all()), "final ids outside codebook"
    check_monotonic(hist)
    fracs = mask_fractions(hist)
    assert fracs[0] == 1.0, f"hist[0] masked fraction {fracs[0]}"
    for a, b in zip(fracs, fracs[1:]):
        assert b <= a + 1e-6, f"masked fraction increased: {a} -> {b}"
    return fracs


@torch.no_grad()
def check_upstream_equivalence(model, uni, mask_id, prompt, seed, steps, cfg):
    """Our rollout vs upstream t2i_generate, same prompt, same generator seed."""
    from models import get_mask_schedule  # upstream, unmodified

    tok = uni.text_tokenizer
    offset = len(tok)
    img_tok = torch.full((1, N_VQ), mask_id, dtype=torch.long, device="cuda")

    ids, am = uni(([prompt], img_tok), "t2i_gen")
    uids, uam = uni(([""], img_tok), "t2i_gen")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    upstream = model.t2i_generate(
        input_ids=ids.clone(), uncond_input_ids=uids.clone(),
        attention_mask=am, uncond_attention_mask=uam,
        temperature=1.0, timesteps=steps, guidance_scale=cfg,
        noise_schedule=get_mask_schedule("cosine"), generator=gen,
        uni_prompting=uni, seq_len=N_VQ, resolution=512,
        mask_token_id=mask_id, codebook_size=8192)

    ours, _, _ = rollout(model, uni, prompt, mask_id, steps=steps, cfg=cfg, seed=seed)
    ours = ours.reshape(1, N_VQ)
    match = torch.equal(ours, upstream.reshape(1, N_VQ))
    diff = int((ours != upstream.reshape(1, N_VQ)).sum())
    return match, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--skip-upstream", action="store_true")
    args = ap.parse_args()

    model, uni, mask_id = setup(seed=42, training=True)
    offset = len(uni.text_tokenizer)
    print(f"loaded; mask_token_id={mask_id} offset={offset}")

    failures = []
    for prompt in PROMPTS:
        final, hist, template = rollout(model, uni, prompt, mask_id,
                                        steps=args.steps, cfg=args.cfg, seed=args.seed)
        fracs = check_structure(final, hist, mask_id, args.steps)
        print(f"[structure] ok  steps={len(hist)} masked@0={fracs[0]:.3f} "
              f"masked@last={fracs[-1]:.3f} final_range=({int(final.min())},{int(final.max())})")

        if not args.skip_upstream:
            match, diff = check_upstream_equivalence(
                model, uni, mask_id, prompt, args.seed, args.steps, args.cfg)
            print(f"[upstream ] {'ok  bit-identical' if match else f'MISMATCH ({diff} positions)'}")
            if not match:
                failures.append(f"upstream mismatch on '{prompt}': {diff} positions differ")

        for view in TeacherView:
            k = min([i for i, f in enumerate(fracs) if f > 0.75], key=lambda i: fracs[i])
            q, counts, scored, info = build_target(
                model, template, hist, k, offset, mask_id, view, probs_at, final,
                gap=2, endpoint_mode="visible")
            early = hist[k]["masked"][0]
            assert q.requires_grad is False, f"{view}: target must be detached"
            assert scored.any(), f"{view}: no scored positions"
            assert not (scored & ~early).any(), f"{view}: scored outside student mask"
            sums = q[scored].sum(-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
                f"{view}: target not normalized (max err {float((sums-1).abs().max())})"
            # every scored position needs a contribution or a fallback
            covered = (counts[scored] > 0)
            n_fallback = int((~covered).sum())
            if n_fallback:
                assert view in (TeacherView.ALL_FUTURE, TeacherView.ENDPOINT_VISIBLE,
                                TeacherView.ENDPOINT_REMASKED), \
                    f"{view}: {n_fallback} scored positions with no contribution"
            print(f"[teacher  ] ok  {view.value:<18} k={k} scored={int(scored.sum())} "
                  f"forwards={info['teacher_forwards']} fallback={n_fallback} "
                  f"entropy={info.get('endpoint_entropy', info.get('fallback_entropy', 'n/a'))}")

    if failures:
        print("\nFAILURES:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nall integration checks passed")


if __name__ == "__main__":
    main()

"""The single rollout implementation, mirrored from upstream `t2i_generate`.

This is the ONLY place that draws an image trajectory. Everything else consumes
`hist` from here.

Layout note: MMaDA's t2i sequence is not [prompt | completion], it is
    [<|t2i|> | text | <eos> | <|soi|> | 1024 image tokens | <|eoi|>]
`ids[:, -(N_VQ+1):-1]` is the image block; codebook ids are stored shifted by
`offset = len(text_tokenizer)`, with `mask_id` passed through unshifted.

The rolling mask count (mask_by_random_topk) names a misleading function: it only
decides which positions are decoded next round. Already-decoded positions are never
re-masked and never change value, so the trajectory is a monotone decoding order and
every hist[t] is a legal on-policy state.

Two subtle points, both verified against upstream by `tests/test_integration.py`:

  * CFG splice. Upstream builds the uncond input as
    `cat([uncond_prefix, input_ids[:, resolution+1:]], 1)` with `uncond_prefix`
    sliced to a HARDCODED `resolution+1 = 513`. For MMaDA's t2i template the text
    portion is exactly 513 positions, so `text_len == 513` here -- but that equality
    is a property of the template, not of the code. We assert it rather than assume it.

  * Temperature annealing. Upstream REASSIGNS its `temperature` parameter inside the
    loop (`temperature = temperature * (1.0 - ratio)`), so the value compounds:
    after step s it is prod_{j<=s} (1 - (j+1)/steps). Writing a fresh local per step
    instead gives a visibly different masking schedule and a different trajectory.
    We default to upstream's compounding behaviour and expose the alternative as
    `anneal="per_step"` for explicit ablation.
"""
import torch

from models import get_mask_schedule
from models.sampling import mask_by_random_topk

from .constants import N_VQ
from .mmada_io import padding_bias

# Upstream slices the uncond prefix at resolution+1; the t2i template puts the
# image block after exactly this many positions.
TEXT_LEN = 513


def _temperature(step, steps, temperature, anneal, running):
    if anneal == "upstream":
        # compounds, matching upstream's in-loop reassignment of its parameter
        return running * (1.0 - (step + 1) / steps)
    if anneal == "per_step":
        # fresh value per step: what the pre-rewrite project code accidentally did
        return temperature * (1.0 - (step + 1) / steps)
    raise ValueError(f"unknown anneal mode: {anneal}")


@torch.no_grad()
def rollout(model, uni, prompt, mask_id, steps=15, cfg=3.5, temperature=1.0,
            seed=0, dev="cuda", anneal="upstream"):
    """Draw one t2i trajectory.

    Returns (final, hist, template):
      final    -- (N_VQ,) long, fully decoded codebook ids in 0..8191
      hist     -- list of `steps` dicts {"state": (1,N_VQ), "masked": (1,N_VQ) bool};
                  hist[t] is the state ENTERING step t, so hist[0] is all-masked and
                  hist[-1] still has positions masked. The fully decoded endpoint is
                  `final`, not hist[-1]["state"].
      template -- (1, L) long, the conditioning text template from UniversalPrompting
    """
    gen = torch.Generator(device=dev).manual_seed(seed)
    img_tok = torch.full((1, N_VQ), mask_id, dtype=torch.long, device=dev)
    ids, am = uni(([prompt], img_tok), "t2i_gen")
    uids, uam = uni(([""], img_tok), "t2i_gen")
    offset = len(uni.text_tokenizer)
    text_len = ids.shape[1] - (N_VQ + 2)
    assert text_len == TEXT_LEN, (
        f"t2i template text portion is {text_len}, expected {TEXT_LEN}; the uncond CFG "
        "splice mirrors upstream's hardcoded resolution+1, so this must hold")

    img = ids[:, -(N_VQ + 1):-1].clone()
    img = torch.where(img == mask_id, mask_id, img - offset)
    schedule = get_mask_schedule("cosine")

    hist = []
    sampled = None
    running_temp = temperature
    for step in range(steps):
        if cfg > 0:
            uncond = torch.cat([uids[:, :TEXT_LEN], ids[:, TEXT_LEN:]], 1)
            bias = (torch.cat([am, uam])[:, :, None]
                    & torch.cat([am, uam])[:, None, :]).bool().unsqueeze(1)
            cond_logits, uncond_logits = model(
                torch.cat([ids, uncond]), attention_bias=bias).logits.chunk(2)
            logits = (1 + cfg) * cond_logits - cfg * uncond_logits
        else:
            logits = model(ids, attention_bias=padding_bias(ids)).logits
        logits = logits[:, -(N_VQ + 1):-1, offset:offset + 8192]
        probs = logits.softmax(-1)

        sampled = torch.multinomial(
            probs.reshape(-1, 8192), 1, generator=gen)[:, 0].view(*logits.shape[:-1])
        unknown = img == mask_id
        sampled = torch.where(unknown, sampled, img)

        ratio = (step + 1) / steps
        selected = torch.gather(probs, -1, sampled.long()[..., None]).squeeze(-1)
        selected = torch.where(unknown, selected, torch.finfo(selected.dtype).max)
        mask_len = (N_VQ * schedule(torch.tensor(ratio))).floor().unsqueeze(0).to(logits.device)
        mask_len = torch.max(torch.tensor([1], device=logits.device),
                             torch.min(unknown.sum(-1, keepdim=True) - 1, mask_len))

        running_temp = _temperature(step, steps, temperature, anneal, running_temp)
        masking = mask_by_random_topk(mask_len, selected, running_temp, generator=gen)

        hist.append({"state": img.clone(), "masked": unknown.clone()})

        ids[:, -(N_VQ + 1):-1] = torch.where(masking, mask_id, sampled + offset)
        img = torch.where(masking, mask_id, sampled)

    # The cosine schedule keeps >=1 position masked on the last step; upstream's return
    # value (`sampled_ids`) is fully concrete, so fill the residue the same way.
    residual = img == mask_id
    if residual.any():
        img = torch.where(residual, sampled, img)
        ids[:, -(N_VQ + 1):-1] = torch.where(residual, sampled + offset,
                                             ids[:, -(N_VQ + 1):-1])
    return img[0], hist, ids


def mask_fractions(hist):
    """Fraction of the 1024 image positions still masked at each recorded state."""
    return [float(h["masked"].float().mean()) for h in hist]


def check_monotonic(hist):
    """Assert the decoding order is monotone: nothing decoded is ever re-masked."""
    for t in range(len(hist) - 1):
        earlier, later = hist[t]["masked"][0], hist[t + 1]["masked"][0]
        assert not (later & ~earlier).any(), f"non-monotonic trajectory at step {t}"
    return True

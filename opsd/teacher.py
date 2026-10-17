"""Teacher target construction, with the teacher view made explicit.

The teacher is always the *same weights* with the LoRA adapter disabled
(`model.disable_adapter()`), under no_grad. What differs between arms is which
state the teacher is queried on. The earlier confusion in this project came from
three definitions sharing one name -- so each is now a `TeacherView`:

  ENDPOINT_VISIBLE    teacher sees the finished image; every scored position is
                      visible to it. MMaDA was only ever trained to predict at
                      masked positions, so a visible-position query is
                      out-of-distribution (measured: argmax == endpoint token only
                      12-19% of the time, entropy 2.8-4.0 vs the student's 5.2-7.1).
  ENDPOINT_REMASKED   same endpoint query, but the scored positions are re-masked
                      before the teacher forward -- a controlled variant of the
                      above, not an obviously better one.
  GAP_N               teacher sees the state n steps after the student's, so
                      positions that were still masked get extra context but most
                      scored positions stay predictive. Scores only the positions
                      still masked at k+n (the shared support set), no fallback.
  ALL_FUTURE          per position, average the teacher's predictive distribution
                      over every future step t > k where that position is STILL
                      masked. Teacher is therefore always queried in-distribution.
                      Costs ~(S - k) teacher forwards per update.

`ALL_FUTURE` is the arm that produced the reported gain; the two ENDPOINT arms are
the "paper-style migration" comparison. Keep them separate when reporting.
"""
import enum

import torch

from .constants import CODEBOOK, N_VQ


class TeacherView(str, enum.Enum):
    ENDPOINT_VISIBLE = "endpoint_visible"
    ENDPOINT_REMASKED = "endpoint_remasked"
    GAP_N = "gap_n"
    ALL_FUTURE = "all_future"


def surviving_mask(early, later):
    """Positions masked at `early` that are *also* still masked at `later`."""
    assert early.dtype == later.dtype == torch.bool
    assert not (later & ~early).any(), "non-monotonic trajectory"
    return early & later


class MaskedProbabilityMean:
    """Streaming, detached per-position probability average over a lifetime.

    Positions only contribute while they are still masked, so each position is
    averaged over its own (random) lifetime rather than a fixed window. That
    normalization is a choice, not a neutral one: a short-lived position gets its
    average from few steps and is weighted equally with a long-lived one in the
    loss. The recorded `future_count_hist` is what exposes it.
    """

    def __init__(self, device, n=N_VQ, vocab=CODEBOOK):
        self.total = torch.zeros(n, vocab, dtype=torch.float32, device=device)
        self.count = torch.zeros(n, dtype=torch.long, device=device)

    @torch.no_grad()
    def add(self, probs, still_masked):
        if (still_masked.dtype != torch.bool or probs.shape != self.total.shape
                or still_masked.shape != self.count.shape):
            raise ValueError("invalid probability or mask shape")
        selected = probs.detach().float()[still_masked]
        if (not torch.isfinite(selected).all() or (selected < 0).any()
                or not torch.allclose(selected.sum(-1), torch.ones_like(selected.sum(-1)),
                                      atol=1e-4, rtol=1e-4)):
            raise ValueError("expected normalized finite probabilities")
        self.total[still_masked] += selected
        self.count[still_masked] += 1

    def _normalized(self, probs, where, what):
        sel = probs[where].detach().float()
        if (not torch.isfinite(sel).all() or (sel < 0).any()
                or not torch.allclose(sel.sum(-1), torch.ones_like(sel.sum(-1)),
                                      atol=1e-4, rtol=1e-4)):
            raise ValueError(f"expected normalized finite {what} probabilities")
        return sel

    @torch.no_grad()
    def finish(self, eligible, fallback=None, require_complete=True):
        """Return (q, counts, scored).

        q        -- (N_VQ, CODEBOOK) detached mean distribution over eligible positions
        counts   -- (N_VQ,) number of contributing future steps
        scored   -- (N_VQ,) bool, positions that received a target

        Positions eligible but with no future contribution take `fallback` if
        supplied; otherwise they are dropped from `scored` when
        `require_complete=False` (the shared-support-set behaviour) and raise.
        """
        valid = eligible & (self.count > 0)
        q = self.total / self.count.clamp_min(1).unsqueeze(-1)
        need = eligible & ~valid
        if need.any():
            if fallback is None:
                if require_complete:
                    raise ValueError(
                        f"{int(need.sum())} positions have no future contribution "
                        "and no fallback was supplied")
            else:
                q[need] = self._normalized(fallback, need, "fallback")
        q[~eligible] = 0.0
        scored = eligible if (need.any() and fallback is not None) else valid
        return q.detach(), self.count.clone(), scored


def endpoint_query_state(final, need, mask_id, mode):
    """Build the teacher's input state for a soft endpoint query on positions `need`."""
    if final.shape != need.shape or need.dtype != torch.bool:
        raise ValueError("invalid endpoint mask")
    if mode not in ("visible", "remasked"):
        raise ValueError(f"unknown endpoint mode: {mode}")
    state = final.detach().clone()
    if mode == "remasked":
        state[need] = mask_id
    return state


@torch.no_grad()
def build_target(model, template, hist, k, offset, mask_id, view,
                 probs_at, final, gap=2, endpoint_mode="visible"):
    """Construct the detached teacher target for the student state hist[k].

    Returns (q, counts, scored, info). `info` records the teacher forward count
    and, where applicable, how the teacher's endpoint view behaved.
    """
    early = hist[k]["masked"][0]
    # `final` is the fully decoded endpoint returned by rollout; hist[-1]["state"] is the
    # state ENTERING the last step and still has positions masked, so it must not be used.
    if (final == mask_id).any():
        raise ValueError("teacher target requires a fully decoded final state")
    acc = MaskedProbabilityMean(early.device)
    info = {"teacher_forwards": 0, "view": str(view)}
    fallback = None

    with model.disable_adapter():
        if view is TeacherView.ALL_FUTURE:
            for t in range(k + 1, len(hist)):
                valid = surviving_mask(early, hist[t]["masked"][0])
                if not valid.any():
                    continue
                probs = probs_at(model, template, hist[t]["state"][0], offset, mask_id)
                acc.add(probs, valid)
                info["teacher_forwards"] += 1
                del probs
            need = early & (acc.count == 0)
            if need.any():
                state = endpoint_query_state(final, need, mask_id, endpoint_mode)
                fallback = probs_at(model, template, state, offset, mask_id)
                info["teacher_forwards"] += 1
                info["fallback_positions"] = int(need.sum())
                info["fallback_entropy"] = float(
                    -(fallback[need] * fallback[need].clamp_min(1e-12).log()).sum(-1).mean())
                del state
            q, counts, scored = acc.finish(early, fallback=fallback)

        elif view is TeacherView.GAP_N:
            t = min(k + gap, len(hist) - 1)
            valid = surviving_mask(early, hist[t]["masked"][0])
            probs = probs_at(model, template, hist[t]["state"][0], offset, mask_id)
            acc.add(probs, valid)
            info["teacher_forwards"] = 1
            info["gap"] = t - k
            del probs
            q, counts, scored = acc.finish(early, require_complete=False)

        elif view in (TeacherView.ENDPOINT_VISIBLE, TeacherView.ENDPOINT_REMASKED):
            mode = "visible" if view is TeacherView.ENDPOINT_VISIBLE else "remasked"
            state = endpoint_query_state(final, early, mask_id, mode)
            fb = probs_at(model, template, state, offset, mask_id)
            info["teacher_forwards"] = 1
            chosen = fb[early]
            info["endpoint_argmax_agreement"] = float(
                (chosen.argmax(-1) == final[early]).float().mean())
            info["endpoint_entropy"] = float(
                -(chosen * chosen.clamp_min(1e-12).log()).sum(-1).mean())
            info["endpoint_sample_probability"] = float(
                chosen.gather(1, final[early, None]).mean())
            del chosen, state
            q, counts, scored = acc.finish(early, fallback=fb)

        else:
            raise ValueError(f"unknown teacher view: {view}")

    info["scored_positions"] = int(scored.sum())
    info["eligible_positions"] = int(early.sum())
    if counts.numel():
        info["future_count_mean"] = float(counts[scored].float().mean()) if scored.any() else 0.0
    return q.detach(), counts, scored, info

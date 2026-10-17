"""Student intermediate-state selection.

dOPSD's rule: among decoding steps whose masked fraction exceeds `tau`, take the
one with the *smallest* such fraction -- the noisiest state that is still
"mostly masked". `fixed` is deterministic and compares arms cleanly; `uniform`
picks uniformly among eligible states and is what the paper-style migration used.
"""
import random

DEFAULT_THRESHOLD = 0.75


def eligible_steps(fracs, threshold=DEFAULT_THRESHOLD):
    return [i for i, f in enumerate(fracs) if f > threshold]


def choose_student(fracs, mode="fixed", threshold=DEFAULT_THRESHOLD, seed=0):
    """Return the index of the student's intermediate state.

    `seed` only affects `uniform`; `fixed` never consumes randomness, so the
    sampler RNG stream is unchanged across selection modes within an update.
    """
    eligible = eligible_steps(fracs, threshold)
    if not eligible:
        raise ValueError(f"no decoding step has masked fraction above {threshold}: "
                         f"max was {max(fracs):.4f}")
    if mode == "fixed":
        return min(eligible, key=lambda i: fracs[i])
    if mode == "least":
        return min(eligible, key=lambda i: fracs[i])
    if mode == "most":
        return max(eligible, key=lambda i: fracs[i])
    if mode == "uniform":
        return random.Random(seed).choice(eligible)
    raise ValueError(f"unknown student selection mode: {mode}")

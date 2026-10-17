"""Objectives on the scored image positions.

Only forward KL is used by the reported arms. The earlier `jsd(beta=...)` helper
was mislabelled -- `beta=1` there computed forward KL(T||S), the opposite of what
dopsd-ref calls `beta=1` -- so no `beta` knob exists here. If you want the other
direction, call `reverse_kl` explicitly and name it in the run config.

  forward_kl(q, student_logits)  = KL(q || p_theta)   -- q is the detached target
  reverse_kl(q, student_logits)  = KL(p_theta || q)

`one_sided` is the OPOD-style ReLU term: constrain only where the teacher likes the
sampled token more than the student does, so the student is not capped by the
teacher. It requires the sampled token ids, so it is opt-in.
"""
import torch


def forward_kl(target_probs, student_logits):
    """Per-position KL(q || p_theta). Returns a (N,) tensor, not reduced."""
    q = target_probs.detach().float()
    return (q * (q.clamp_min(1e-12).log() - student_logits.float().log_softmax(-1))).sum(-1)


def reverse_kl(target_probs, student_logits):
    """Per-position KL(p_theta || q). Returns a (N,) tensor, not reduced."""
    q = target_probs.detach().float()
    log_p = student_logits.float().log_softmax(-1)
    return (log_p.exp() * (log_p - q.clamp_min(1e-12).log())).sum(-1)


def one_sided(target_probs, student_logits, sampled_tokens, eps=1e-12):
    """Per-position ReLU(log q(x_hat) - log p_theta(x_hat)).

    Nonzero only where the teacher assigns the *sampled* token a higher probability
    than the student does; the student is otherwise unconstrained at that position.
    """
    q = target_probs.detach().float()
    log_q = q.clamp_min(eps).log().gather(1, sampled_tokens[:, None]).squeeze(1)
    log_p = student_logits.float().log_softmax(-1).gather(
        1, sampled_tokens[:, None]).squeeze(1)
    return torch.relu(log_q - log_p)


def entropy(probs, eps=1e-12):
    """Mean per-position entropy of a probability tensor, for monitoring."""
    p = probs.detach().float()
    return float(-(p * p.clamp_min(eps).log()).sum(-1).mean())

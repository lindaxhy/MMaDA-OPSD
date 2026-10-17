"""Unit tests for the pure-logic pieces. Run on CPU: `python -m pytest tests/ -q`
or `python -m unittest discover -s tests -p 'test_*.py' -v`.

These pin the invariants that earlier debugging found were easy to break:
  - the teacher target is a normalized distribution and is detached
  - every scored position either has >=1 future contribution or a fallback
  - selection never depends on the rollout RNG stream
  - forward KL is zero at q == p and asymmetric in the right direction
"""
import unittest

import torch

from opsd.loss import entropy, forward_kl, one_sided, reverse_kl
from opsd.student import choose_student, eligible_steps
from opsd.teacher import MaskedProbabilityMean, TeacherView, endpoint_query_state, surviving_mask

DEV = "cpu"
N, VOCAB = 8, 5


def onehot(idx, n=N, vocab=VOCAB):
    t = torch.zeros(n, vocab)
    t[torch.arange(n), idx] = 1.0
    return t


class TestSurvivingMask(unittest.TestCase):
    def test_intersection(self):
        # monotone: positions only ever leave the masked set
        early = torch.tensor([1, 1, 1, 1], dtype=torch.bool)
        later = torch.tensor([1, 0, 0, 1], dtype=torch.bool)
        got = surviving_mask(early, later)
        self.assertEqual(got.tolist(), [True, False, False, True])

    def test_intersection_excludes_positions_decoded_before_early(self):
        early = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        later = torch.tensor([1, 1, 1, 1], dtype=torch.bool)  # INVALID: 1,3 re-masked
        with self.assertRaises(AssertionError):
            surviving_mask(early, later)

    def test_intersection_is_early(self):
        early = torch.tensor([1, 1], dtype=torch.bool)
        later = torch.tensor([1, 0], dtype=torch.bool)
        self.assertEqual(surviving_mask(early, later).tolist(), [True, False])

    def test_rejects_non_monotonic(self):
        early = torch.tensor([0, 0], dtype=torch.bool)
        later = torch.tensor([1, 0], dtype=torch.bool)  # became masked again
        with self.assertRaises(AssertionError):
            surviving_mask(early, later)


class TestMaskedProbabilityMean(unittest.TestCase):
    def test_average_over_lifetime(self):
        acc = MaskedProbabilityMean(DEV, n=2, vocab=3)
        m1 = torch.tensor([True, False])
        m2 = torch.tensor([True, True])
        acc.add(torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), m1)
        acc.add(torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]), m2)
        q, counts, scored = acc.finish(torch.tensor([True, True]))
        self.assertEqual(counts.tolist(), [2, 1])
        torch.testing.assert_close(q[0], torch.tensor([0.5, 0.5, 0.0]))
        torch.testing.assert_close(q[1], torch.tensor([0.0, 1.0, 0.0]))
        self.assertTrue(scored.all())

    def test_missing_contribution_requires_fallback(self):
        acc = MaskedProbabilityMean(DEV, n=2, vocab=3)
        acc.add(torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                torch.tensor([True, False]))
        with self.assertRaises(ValueError):
            acc.finish(torch.tensor([True, True]))

    def test_missing_contribution_dropped_when_not_required(self):
        acc = MaskedProbabilityMean(DEV, n=2, vocab=3)
        acc.add(torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                torch.tensor([True, False]))
        _, counts, scored = acc.finish(torch.tensor([True, True]), require_complete=False)
        self.assertEqual(scored.tolist(), [True, False])
        self.assertEqual(counts.tolist(), [1, 0])

    def test_fallback_fills_and_is_detached(self):
        acc = MaskedProbabilityMean(DEV, n=2, vocab=3)
        acc.add(torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                torch.tensor([True, False]))
        fb = torch.tensor([[0.0, 0.0, 1.0], [0.5, 0.5, 0.0]], requires_grad=True)
        q, _, scored = acc.finish(torch.tensor([True, True]), fallback=fb)
        self.assertTrue(scored.all())
        torch.testing.assert_close(q[1], torch.tensor([0.5, 0.5, 0.0]))
        self.assertFalse(q.requires_grad)

    def test_rejects_unnormalized(self):
        acc = MaskedProbabilityMean(DEV, n=2, vocab=3)
        bad = torch.tensor([[0.7, 0.0, 0.0], [0.0, 0.0, 0.1]])
        with self.assertRaises(ValueError):
            acc.add(bad, torch.tensor([True, True]))

    def test_ineligible_positions_zeroed(self):
        acc = MaskedProbabilityMean(DEV, n=2, vocab=3)
        acc.add(torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                torch.tensor([True, True]))
        q, _, _ = acc.finish(torch.tensor([True, False]))
        torch.testing.assert_close(q[1], torch.zeros(3))


class TestEndpointQueryState(unittest.TestCase):
    def test_visible_keeps_endpoint_ids(self):
        final = torch.tensor([3, 4, 5])
        need = torch.tensor([True, False, True])
        got = endpoint_query_state(final, need, mask_id=99, mode="visible")
        self.assertEqual(got.tolist(), [3, 4, 5])

    def test_remasked_blanks_only_need(self):
        final = torch.tensor([3, 4, 5])
        need = torch.tensor([True, False, False])
        got = endpoint_query_state(final, need, mask_id=99, mode="remasked")
        self.assertEqual(got.tolist(), [99, 4, 5])

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            endpoint_query_state(torch.tensor([1]), torch.tensor([True]), 0, "bogus")


class TestStudentSelection(unittest.TestCase):
    FRACS = [1.0, 0.9, 0.8, 0.6, 0.3, 0.0]

    def test_fixed_is_deterministic_and_ignores_seed(self):
        a = choose_student(self.FRACS, "fixed", 0.75, seed=1)
        b = choose_student(self.FRACS, "fixed", 0.75, seed=999)
        self.assertEqual(a, b)
        self.assertEqual(a, 2)  # smallest fraction still above 0.75

    def test_uniform_only_picks_eligible(self):
        for seed in range(50):
            k = choose_student(self.FRACS, "uniform", 0.75, seed=seed)
            self.assertIn(k, eligible_steps(self.FRACS, 0.75))

    def test_raises_when_nothing_eligible(self):
        with self.assertRaises(ValueError):
            choose_student([0.5, 0.2], "fixed", 0.75, seed=0)


class TestLosses(unittest.TestCase):
    def test_forward_kl_zero_at_match(self):
        q = torch.softmax(torch.randn(N, VOCAB), -1)
        logits = q.clamp_min(1e-12).log()
        torch.testing.assert_close(forward_kl(q, logits), torch.zeros(N), atol=1e-5, rtol=1e-5)

    def test_forward_kl_is_not_reverse_kl(self):
        q = torch.softmax(torch.randn(N, VOCAB), -1)
        logits = torch.randn(N, VOCAB)
        self.assertGreater((forward_kl(q, logits) - reverse_kl(q, logits)).abs().sum(), 0.0)

    def test_forward_kl_matches_torch_reference(self):
        q = torch.softmax(torch.randn(N, VOCAB), -1)
        logits = torch.randn(N, VOCAB)
        ref = torch.nn.functional.kl_div(logits.log_softmax(-1), q,
                                         reduction="none", log_target=False).sum(-1)
        torch.testing.assert_close(forward_kl(q, logits), ref, atol=1e-5, rtol=1e-5)

    def test_one_sided_is_zero_when_student_agrees(self):
        q = onehot(torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]))
        logits = q.clamp_min(1e-12).log() * 10  # student matches exactly
        sampled = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
        torch.testing.assert_close(one_sided(q, logits, sampled), torch.zeros(N))

    def test_one_sided_positive_when_teacher_prefers_sampled(self):
        q = onehot(torch.tensor([0] * N))
        logits = torch.zeros(N, VOCAB)  # uniform student
        sampled = torch.zeros(N, dtype=torch.long)
        self.assertTrue((one_sided(q, logits, sampled) > 0).all())

    def test_entropy_bounds(self):
        self.assertAlmostEqual(entropy(torch.full((N, VOCAB), 1.0 / VOCAB)), 1.6094, places=3)
        self.assertAlmostEqual(entropy(onehot(torch.zeros(N, dtype=torch.long))), 0.0, places=5)


class TestTeacherViewEnum(unittest.TestCase):
    def test_all_views_have_string_values(self):
        self.assertEqual(TeacherView.ALL_FUTURE.value, "all_future")
        self.assertEqual(len(set(v.value for v in TeacherView)), len(list(TeacherView)))


if __name__ == "__main__":
    unittest.main()

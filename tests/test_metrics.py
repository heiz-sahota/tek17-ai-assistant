"""Unit tests for eval/metrics.py. Run: pytest -q"""
import pytest

from eval.metrics import first_hit_rank, hit_rate_at_k, mrr

RANKS = [1, 3, None, 2]


def test_hit_rate_at_k():
    assert hit_rate_at_k(RANKS, 1) == 0.25
    assert hit_rate_at_k(RANKS, 3) == 0.75
    assert hit_rate_at_k([], 5) == 0.0


def test_mrr():
    assert mrr(RANKS) == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)
    assert mrr(RANKS, k=2) == pytest.approx((1 + 0 + 0 + 1 / 2) / 4)


def test_first_hit_rank():
    assert first_hit_rank(["a", "b", "c"], lambda x: x == "b") == 2
    assert first_hit_rank(["a"], lambda x: x == "z") is None
"""
eval/metrics.py: retrieval metrics, plain Python, no Azure calls.

Every function takes `ranks`: for each question, the 1-based position of the
FIRST correct chunk in the results, or None if no correct chunk was returned.

    ranks = [1, 3, None, 2]
    hit_rate_at_k(ranks, 1) -> 0.25   only the first question had it at the top
    hit_rate_at_k(ranks, 3) -> 0.75   three of four had it in the top 3
    mrr(ranks)              -> (1 + 1/3 + 0 + 1/2) / 4 = 0.458

HIT RATE AT K  (also called recall at k when there is one correct section)
    The share of questions where a correct chunk is in the top k.
    Use the k you actually send to the model: if the prompt gets 5 chunks,
    hit rate at 5 says how often the model even has a chance to answer.

MRR, MEAN RECIPROCAL RANK
    The average of 1/rank. It rewards finding the right chunk EARLY:
    rank 1 = 1.0, rank 2 = 0.5, rank 3 = 0.33, missing = 0.
    Two systems with the same hit rate at 5 can differ a lot in MRR.

Unit-tested in tests/test_metrics.py, so the numbers you quote are right.
"""


def hit_rate_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None], k: int | None = None) -> float:
    """Mean reciprocal rank, optionally only counting ranks up to k."""
    if not ranks:
        return 0.0
    total = 0.0
    for r in ranks:
        if r is not None and (k is None or r <= k):
            total += 1 / r
    return total / len(ranks)


def first_hit_rank(results: list, is_correct) -> int | None:
    """1-based rank of the first result where is_correct(result) is true."""
    for position, result in enumerate(results, start=1):
        if is_correct(result):
            return position
    return None
import numpy as np

from tests.notebook_code import batch_metrics, topk_from_scores, user_metrics


def _rel(u):
    return u


def test_user_metrics_basic():
    top = np.array([3, 5, 7, 1, 0, 2, 4, 6, 8, 9], dtype=np.int64)
    rel = np.array([3, 9, 11])
    m = user_metrics(top, rel, 10)
    # hits at ranks 1 (3) and 10 (9)
    assert abs(m["hr"] - 2 / 3) < 1e-9
    assert abs(m["mrr"] - 1.0) < 1e-9
    assert abs(m["recall"] - 2 / 3) < 1e-9


def test_topk_order():
    scores = np.array([[1.0, 3.0, 2.0, 0.5], [4.0, 1.0, 1.0, 1.5]], dtype=np.float32)
    top = topk_from_scores(scores, 2)
    assert top.shape == (2, 2)
    # check the returned rows are descending by score
    for row in range(scores.shape[0]):
        vals = scores[row, top[row]]
        assert np.all(np.diff(vals) <= 1e-6)
    assert set(top[0]) == {1, 2}
    assert set(top[1]) == {0, 3}


def test_batch_metrics():
    scores = np.array([
        [0.9, 0.8, 0.1, 0.0],
        [0.1, 0.9, 0.0, 0.8],
    ], dtype=np.float32)
    top = topk_from_scores(scores, 2)
    # user 0 top = [0, 1] (item 1 at rank 2 = first hit); user 1 top = [1, 3] (item 3 at rank 2)
    rel = {0: np.array([1]), 1: np.array([2, 3])}
    m = batch_metrics(top, rel, np.array([0, 1], dtype=np.int32), 2)
    assert abs(m["hr"] - 0.75) < 1e-9
    assert abs(m["mrr"] - 0.5) < 1e-9


def test_no_relevant():
    m = user_metrics(np.arange(10), np.array([50]), 10)
    assert all(v == 0.0 for v in m.values())

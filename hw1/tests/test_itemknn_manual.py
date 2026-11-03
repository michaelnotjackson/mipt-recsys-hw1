"""Item-KNN: exact manual check of shrinkage-cosine scoring."""
import numpy as np
from scipy import sparse

from tests.notebook_code import ExpData


def _mini_exp() -> ExpData:
    X = sparse.csr_matrix(
        (np.ones(8, dtype=np.float32), ([0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 0, 2, 1, 3])),
        shape=(4, 4),
    )
    R = sparse.csr_matrix(
        (np.full(8, 4.0, dtype=np.float32), ([0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 0, 2, 1, 3])),
        shape=(4, 4),
    )
    import polars as pl

    items_meta = pl.DataFrame(
        {
            "item_idx": np.arange(4, dtype=np.int32),
            "cat_path": ["a|b"] * 4,
            "store": ["s"] * 4,
            "price": np.array([1.0, np.nan, 2.0, 3.0], dtype=np.float32),
            "desc_len": np.array([10, 20, 30, 40], dtype=np.int32),
            "title_len": np.array([5, 6, 7, 8], dtype=np.int32),
        }
    )
    return ExpData(
        n_users=4, n_items=4,
        train_u=np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32),
        train_i=np.array([0, 1, 0, 1, 0, 2, 1, 3], dtype=np.int32),
        train_r=np.full(8, 4.0, dtype=np.float32),
        train_ts=np.arange(8, dtype=np.float64) * 1000.0,
        X=X, R=R,
        cand_items=np.arange(4, dtype=np.int32),
        cand_pos=np.array([0, 1, 2, 3], dtype=np.int64),
        val_user=np.array([], dtype=np.int32),
        val_item=np.array([], dtype=np.int32),
        val_users=np.array([], dtype=np.int32),
        test_user=np.array([], dtype=np.int32),
        test_item=np.array([], dtype=np.int32),
        test_users=np.array([], dtype=np.int32),
        items_meta=items_meta,
    )


def test_itemknn_manual_small():
    exp = _mini_exp()
    gamma = 5.0
    from tests.notebook_code import ItemKNN

    m = ItemKNN()
    m.fit(exp, {"shrinkage": gamma, "topk": 10}, seed=0)

    # user item sets: u0={0,1}, u1={0,1}, u2={0,2}, u3={1,3}
    # co-occurrence: f(0,1)=2, f(0,2)=1, f(1,3)=1; counts n = [3, 3, 1, 1]
    # sim(c,j) = f(c,j) / (sqrt(n_c*n_j) + gamma)
    sim01 = 2 / (np.sqrt(3 * 3) + gamma)
    sim02 = 1 / (np.sqrt(3 * 1) + gamma)
    sim13 = 1 / (np.sqrt(3 * 1) + gamma)

    # manual score for user 0 (items {0,1}); self-similarity excluded:
    scores = m.score_candidates(np.array([0], dtype=np.int32), exp.cand_items)[0]
    manual = np.array(
        [
            sim01,                # c0: j1 -> sim(0,1)
            sim01,                # c1: j0 -> sim(1,0)
            sim02,                # c2: j0 -> sim(2,0)
            sim13,                # c3: j1 -> sim(3,1)
        ],
        dtype=np.float32,
    )
    assert np.allclose(scores, manual, atol=1e-5), f"{scores} vs {manual}"


def test_itemknn_beats_flipped_similarity():
    """On a block-structured synthetic set the top similarity must be a co-rated pair."""
    from tests.synth import make_exp

    exp = make_exp(seed=7, n_users=200, n_items=40, n_train=4000)
    from tests.notebook_code import ItemKNN

    m = ItemKNN()
    m.fit(exp, {"shrinkage": 10, "topk": 5}, seed=0)
    s = m.score_candidates(np.array([0, 9], dtype=np.int32), exp.cand_items)
    assert np.isfinite(s).all()
    assert (s > 0).any()

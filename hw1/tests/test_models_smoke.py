"""End-to-end smoke: fit all models on synthetic data and check the interface."""
import numpy as np
import pytest

from tests.synth import make_exp  # noqa: I001

MODELS = {
    "itemknn": {"shrinkage": 10, "topk": 20},
    "ease": {"lam": 5.0},
    "als": {"f": 4, "iterations": 2, "lam": 0.05},
    "fm": {"dim": 4, "epochs": 2, "lr": 0.05, "feature_set": "full"},
    "timesvdpp": {"f": 4, "epochs": 2, "lr": 0.01, "bin_days": 3650},
    "popularity": {},
    "random": {},
}
CLS = {
    "itemknn": "ItemKNN", "ease": "EASE", "als": "ALS", "fm": "FM",
    "timesvdpp": "TimeSVDPP", "popularity": "Popularity", "random": "RandomModel",
}


@pytest.mark.parametrize("name", list(MODELS.keys()))
def test_model_smoke(name):
    import tests.notebook_code as M

    exp = make_exp(seed=1)
    model = getattr(M, CLS[name])()
    model.fit(exp, MODELS[name], seed=0)

    uids = np.array([0, 1, 2], dtype=np.int32)
    scores = model.score_candidates(uids, exp.cand_items)
    assert scores.shape == (3, len(exp.cand_items))
    assert np.isfinite(scores).all()
    assert scores.dtype == np.float32

    # eval interface returns sane metrics
    from tests.notebook_code import compress_rel_sets
    from tests.notebook_code import EvalSet, build_eval_set, evaluate_model, topk_for_user

    es = build_eval_set(exp.test_user, exp.test_item, exp)
    res = evaluate_model(model, exp, es, k=5, chunk_users=16)
    m = res.metrics
    assert res.n_users > 0
    assert 0.0 <= m["hr"] <= 1.0 and 0.0 <= m["ndcg"] <= 1.0
    assert np.isfinite(res.top_items).all()
    t = topk_for_user(model, exp, int(es.users[0]), k=5)
    assert all(it in set(exp.cand_items.astype(np.int64)) for it in t)

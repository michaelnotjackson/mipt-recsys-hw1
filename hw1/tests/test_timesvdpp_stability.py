"""timeSVD++: finiteness + convergence of the clamped sum-semantics SGD.

The training regime mirrors the production config: several batches per epoch,
moderate lr, prediction clipping. On tiny data the per-epoch SUM steps are
amplified (fewer samples per row), so the test enqueue a small batch size.
"""
import numpy as np

from tests.notebook_code import TimeSVDPP
from tests.synth import make_exp


def test_timesvdpp_stable_and_converging():
    exp = make_exp(seed=11, n_users=300, n_items=50, n_train=6000)
    m = TimeSVDPP()
    m.fit(exp, {"f": 8, "lr": 0.02, "epochs": 25, "bin_days": 365, "batch_size": 1024}, seed=0)
    h = m.history()["rmse"]
    assert np.isfinite(np.array(h, dtype=np.float64)).all(), h
    assert h[-1] < h[0], h  # training reduces the residual error
    for p in (m._bi, m._bu, m._bi_b, m._bu_b):
        v = p.numpy()
        assert np.isfinite(v).all()
        assert np.abs(v).max() < 50.0
    # mu is close to the mean rating (the mu sign was fixed; no systematic drift)
    assert abs(float(m._mu) - float(exp.train_r.mean())) < 0.5


def test_timesvdpp_no_divergence_hot_items():
    """A few very popular items must not blow up the parameters."""
    from scipy import sparse

    exp = make_exp(seed=3, n_users=500, n_items=30, n_train=8000)
    # make some items extremely hot: 40% of all interactions on 3 items
    n_hot = int(0.4 * len(exp.train_u))
    hot = np.arange(n_hot)
    i2 = exp.train_i.copy()
    m_hot = n_hot - (n_hot % 3)
    i2[hot[:m_hot]] = np.resize(np.array([0, 1, 2], dtype=np.int32), m_hot)
    i2[hot[m_hot:]] = 0
    exp.train_i = i2
    exp.X = sparse.csr_matrix(
        (np.ones(len(i2), dtype=np.float32), (exp.train_u, i2)),
        shape=(exp.n_users, exp.n_items),
    )
    m = TimeSVDPP()
    m.fit(exp, {"f": 8, "lr": 0.02, "epochs": 20, "bin_days": 365, "batch_size": 1024}, seed=0)
    h = m.history()["rmse"]
    assert np.isfinite(np.array(h, dtype=np.float64)).all(), h
    q_absmax = float(np.abs(m._Q.numpy()).max())
    assert q_absmax < 20.0, q_absmax

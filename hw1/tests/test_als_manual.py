"""ALS: naive per-row reference implementation vs the vectorized model."""
import numpy as np

from tests.notebook_code import ALS
from tests.test_itemknn_manual import _mini_exp


def _naive_als(exp, f, iterations, lam, implicit, alpha, seed):
    u, i, r = exp.train_u, exp.train_i, exp.train_r.astype(np.float64)
    U, I = exp.n_users, exp.n_items
    if implicit:
        w = 1.0 + alpha * r
        t = np.ones_like(r)
    else:
        mu = float(r.mean())
        n_i = np.bincount(i, minlength=I).astype(np.float64)
        b_i = np.bincount(i, weights=r - mu, minlength=I) / (n_i + 20.0)
        b_u = np.bincount(u, weights=r - mu - b_i[i], minlength=U) / (
            np.bincount(u, minlength=U) + 20.0
        )
        e = r - mu - b_u[u] - b_i[i]
        w, t = np.ones_like(r), e

    rng = np.random.default_rng(seed)
    P = rng.standard_normal((U, f)) * 0.05
    Q = rng.standard_normal((I, f)) * 0.05
    for _ in range(iterations):
        for uu in range(U):
            m = i[u == uu]
            if len(m) == 0:
                continue
            Z = Q[m] * np.sqrt(w[u == uu])[:, None]
            A = Z.T @ Z + lam * np.eye(f)
            b = Z.T @ (t[u == uu] * np.sqrt(w[u == uu]))
            P[uu] = np.linalg.solve(A, b)
        for jj in range(I):
            m = u[i == jj]
            if len(m) == 0:
                continue
            Z = P[m] * np.sqrt(w[i == jj])[:, None]
            A = Z.T @ Z + lam * np.eye(f)
            b = Z.T @ (t[i == jj] * np.sqrt(w[i == jj]))
            Q[jj] = np.linalg.solve(A, b)
    return P, Q


def test_als_explicit_matches_naive():
    exp = _mini_exp()  # 4 users, 4 items, 8 ratings
    m = ALS()
    m.fit(exp, {"f": 4, "iterations": 3, "lam": 1.0, "lam_b": 20.0, "implicit": False}, seed=3)
    P_ref, Q_ref = _naive_als(exp, 4, 3, 1.0, False, 0.0, 3)
    assert np.allclose(m._P, P_ref, atol=1e-3)
    assert np.allclose(m._Q, Q_ref, atol=1e-3)
    # scores include biases
    s = m.score_candidates(np.array([0], dtype=np.int32), exp.cand_items)[0]
    manual = (
        m._mu + m._b_u[0] + m._b_i[exp.cand_items] + P_ref[0] @ Q_ref[exp.cand_items].T
    )
    assert np.allclose(s, manual, atol=2e-3)


def test_als_implicit_matches_naive():
    exp = _mini_exp()
    m = ALS()
    m.fit(exp, {"f": 4, "iterations": 3, "lam": 1.0, "implicit": True, "alpha": 2.0}, seed=3)
    P_ref, Q_ref = _naive_als(exp, 4, 3, 1.0, True, 2.0, 3)
    assert np.allclose(m._P, P_ref, atol=1e-3)
    assert np.allclose(m._Q, Q_ref, atol=1e-3)
    s = m.score_candidates(np.array([0], dtype=np.int32), exp.cand_items)[0]
    assert np.allclose(s, P_ref[0] @ Q_ref[exp.cand_items].T, atol=2e-3)

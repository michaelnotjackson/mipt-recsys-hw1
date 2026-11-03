"""timeSVD++: manual formula check of the scoring path."""
import numpy as np

from tests.notebook_code import TimeSVDPP
from tests.synth import make_exp


def test_timesvdpp_scoring_formula():
    exp = make_exp(seed=5)
    m = TimeSVDPP()
    m.fit(exp, {"f": 4, "epochs": 2, "lr": 0.01, "bin_days": 365, "batch_size": 256}, seed=0)

    t_rec = float(exp.train_ts.max())  # in-span timestamp: bins exist
    m.set_t_rec(t_rec)
    uids = np.array([0, 5, 17], dtype=np.int32)
    cands = exp.cand_items[:8].astype(np.int64)

    scores = m.score_candidates(uids, cands)

    # manual formula (dev in years scale, matching the training code)
    bin_rec = int((t_rec - m._t_min) / m._bin_len)
    prof_np = m._profile().numpy()
    manual = np.zeros_like(scores)
    for b, uid in enumerate(uids):
        yr = (t_rec - m._anchor[uid]) / (365.25 * 86_400.0)
        dev = np.sign(yr) * abs(yr) ** (2.0 / 3.0)
        if not np.isfinite(dev):
            dev = 0.0
        for c_col, item in enumerate(cands):
            v = float(m._mu)
            v += float(m._bi[item])
            if 0 <= bin_rec < m._nb:
                v += float(m._bi_b[item * m._nb + bin_rec])
                v += float(m._bu_b[uid * m._nb + bin_rec])
            v += float(m._bu[uid]) + float(m._au[uid]) * float(dev)
            v += float(np.dot(prof_np[uid], m._Q[item]))
            manual[b, c_col] = v
    assert np.allclose(scores, manual, atol=1e-2, rtol=1e-2)

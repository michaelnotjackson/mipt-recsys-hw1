"""FM: decomposed scoring path must match the module forward pass exactly."""
import numpy as np
import torch

import tests.notebook_code as fmod
from tests.notebook_code import FM, _FMNet
from tests.synth import make_exp  # noqa: I001


def test_fm_forward_vs_scoring():
    exp = make_exp(seed=2)
    ft = fmod.build_feature_tables(exp)
    fields = ft.user_side + ft.item_side
    net = _FMNet(fields, ft.vocabs, dim=8)

    fm = FM()
    fm._net = net
    fm._ft = ft
    fm._clear_item_cache()
    fm._t_rec = float(exp.train_ts.max())

    uids = np.array([0, 3, 7, 42], dtype=np.int32)
    items = exp.cand_items[:6]
    ts = np.full(len(uids), float(exp.train_ts.max()))

    # forward over full batch (each user x each item)
    B = len(uids)
    ub = ft.batch_user(uids, ts)  # [B, Fu]
    ib = ft.batch_item(items)  # [C, Fi]
    ids_full = np.concatenate([np.repeat(ub, len(items), axis=0), np.tile(ib, (B, 1))], axis=1)
    with torch.no_grad():
        preds = net(torch.from_numpy(ids_full)).numpy().reshape(B, len(items))

    scores = fm.score_candidates(uids, items)
    assert np.allclose(preds, scores, atol=1e-3, rtol=1e-3)


def test_fm_ids_only_fields():
    exp = make_exp(seed=3)
    ft = fmod.build_feature_tables(exp).with_fields(["user"], ["item"])
    assert ft.vocabs["user"] == exp.n_users
    assert list(ft.item_side) == ["item"]

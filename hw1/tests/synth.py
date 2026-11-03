"""Shared synthetic ExpData builders for tests."""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import sparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.notebook_code import ExpData, build_candidates  # noqa: E402


def make_exp(seed: int = 0, n_users: int = 120, n_items: int = 60, n_train: int = 1500,
             n_eval_per_user: int = 2, cand_min: int = 1, cand_max: int | None = None) -> ExpData:
    rng = np.random.default_rng(seed)
    pop = (1.0 / (np.arange(1, n_items + 1) ** 0.3))  # popular items more frequent
    pop = pop / pop.sum()

    def sample_rows(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u = rng.integers(0, n_users, size=n).astype(np.int32)
        i = rng.choice(n_items, size=n, p=pop).astype(np.int32)
        r = rng.integers(1, 6, size=n).astype(np.float32)
        ts = np.sort(rng.uniform(1_500_000_000, 1_600_000_000, size=n).astype(np.float64))
        return u, i, r, ts

    tu, ti, tr, tts = sample_rows(n_train)
    vu, vi, _, _ = sample_rows(n_eval_per_user * n_users)
    teu, tei, _, _ = sample_rows(n_eval_per_user * n_users)

    X = sparse.csr_matrix((np.ones(n_train, dtype=np.float32), (tu.astype(np.int64), ti.astype(np.int64))),
                          shape=(n_users, n_items))
    R = sparse.csr_matrix((tr, (tu.astype(np.int64), ti.astype(np.int64))), shape=(n_users, n_items))

    cand = build_candidates(ti.astype(np.int64), n_items, cand_min, cand_max or n_items)
    cand_pos = np.full(n_items, -1, dtype=np.int64)
    cand_pos[cand.astype(np.int64)] = np.arange(len(cand), dtype=np.int64)

    # users/items meta (str fields compatible with features.build_feature_tables)
    items_meta = pl.DataFrame({
        "item_idx": np.arange(n_items, dtype=np.int32),
        "item_id_raw": [f"{x:05d}" for x in range(n_items)],
        "cat_main": [f"cat{k % 5}" for k in range(n_items)],
        "cat_path": [f"cat{k % 5}|sub{k % 7}" for k in range(n_items)],
        "title": [f"Product {k}" for k in range(n_items)],
        "store": [f"store{k % 10}" if k % 4 else "" for k in range(n_items)],
        "price": np.where(np.arange(n_items) % 7 == 0, np.nan, (rng.integers(5, 100, size=n_items) / 10).astype(np.float32)).astype(np.float32),
        "desc_len": rng.integers(0, 500, size=n_items).astype(np.int32),
        "title_len": rng.integers(5, 50, size=n_items).astype(np.int32),
    })

    return ExpData(
        n_users=n_users, n_items=n_items,
        train_u=tu, train_i=ti, train_r=tr, train_ts=tts,
        X=X, R=R,
        cand_items=cand, cand_pos=cand_pos,
        val_user=vu, val_item=vi, val_users=np.unique(vu).astype(np.int32),
        test_user=teu, test_item=tei, test_users=np.unique(teu).astype(np.int32),
        items_meta=items_meta,
    )

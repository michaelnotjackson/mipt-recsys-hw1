import numpy as np
import polars as pl

from tests.notebook_code import filter_core, sample_users_to_cap


def test_filter_core_drops_rare():
    df = pl.DataFrame({
        "user_id_raw": ["u1", "u1", "u2", "u3", "u3", "u3", "u4"],
        "item_id_raw": ["a", "b", "a", "a", "b", "c", "d"],
        "rating": [5.0] * 7,
        "ts": [1.0] * 7,
    })
    out = filter_core(df, 2)
    # u1 (2), u3 (3) survive; u2, u4 dropped; items a (3), b (2) survive; c, d dropped
    users = set(out["user_id_raw"].to_list())
    items = set(out["item_id_raw"].to_list())
    assert users == {"u1", "u3"} and items == {"a", "b"}


def test_sample_users_to_cap_deterministic():
    df = pl.DataFrame({
        "user_id_raw": [f"u{i:03d}" for i in range(100) for _ in range(10)],  # 10 each
        "item_id_raw": [f"i{(i * 10 + j) % 50:03d}" for i in range(100) for j in range(10)],
        "rating": [5.0] * 1000,
        "ts": [float(i) for i in range(1000)],
    })
    out, info = sample_users_to_cap(df, cap=11_000, seed=42)
    assert info["sampling"] == "not-needed"
    out2, info2 = sample_users_to_cap(df, cap=550, seed=42)
    assert out2.height <= 550
    # same seed => same users
    out3, _ = sample_users_to_cap(df, cap=550, seed=42)
    assert out2.equals(out3)
    assert set(out2["user_id_raw"].unique().to_list()) == set(out3["user_id_raw"].unique().to_list())
    # every sampled user keeps all interactions
    counts = out2.group_by("user_id_raw").len()
    assert (counts["len"] == 10).all()

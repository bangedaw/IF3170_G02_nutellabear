"""Tes preprocessing pada data asli: `python -m pytest tests/test_preprocessing.py` dari root repo."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import (TARGET, CategoricalEncoder, drop_duplicates_and_invalid, handle_imbalance,  # noqa: E402
                           make_views, numeric_pipeline, split_train_val)

DATA = ROOT / "T2-IF3170-2026-2027"
df = pd.read_csv(DATA / "train.csv", dtype={"id": str})
test = pd.read_csv(DATA / "test.csv", dtype={"id": str})
tr, va = split_train_val(df)
tr_clean, report = drop_duplicates_and_invalid(tr)


def test_split_is_stratified():
    assert len(tr) + len(va) == len(df)
    assert abs(tr[TARGET].mean() - va[TARGET].mean()) < 0.002


def test_cleaning_only_drops_rows_and_reports():
    assert report["rows_out"] == len(tr_clean) < len(tr)
    assert report["rows_in"] - report["rows_out"] == (report["conflicting_duplicates"] + report["duplicates"]
                                                     + report["invalid"])
    assert not tr_clean.drop(columns="id").duplicated().any()


def test_views_shape_and_no_leak():
    views = make_views(tr_clean, va, test)
    for name in ("numeric", "tree"):
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = views[name]
        assert len(X_tr) == len(tr_clean) and len(X_va) == len(va) and len(X_te) == len(test) == 6844
        assert list(X_tr.columns) == list(X_te.columns)
        assert y_te is None and "id" not in X_tr.columns
    (X_tr, _), (X_va, _), (X_te, _) = views["numeric"]
    assert not X_tr.isna().any().any() and not X_va.isna().any().any() and not X_te.isna().any().any()
    assert np.allclose(X_tr.mean(), 0, atol=1e-8)
    assert ((X_tr.std(ddof=0) - 1).abs() < 1e-8).all()  # tidak ada kolom yang jadi konstan


def test_unknown_category_does_not_fail():
    odd = va.head(3).copy()
    odd["room_category"] = "Type Z"
    enc = CategoricalEncoder().fit(tr_clean)
    assert not enc.transform(odd).filter(like="room_category=").to_numpy().any()
    X, _ = numeric_pipeline().fit(tr_clean).transform(odd)  # pipeline penuh tetap jalan
    assert not X.isna().any().any()


def test_imbalance_only_balances():
    X, y = handle_imbalance(np.arange(10)[:, None], np.array([0] * 7 + [1] * 3))
    assert np.bincount(y).tolist() == [7, 7]

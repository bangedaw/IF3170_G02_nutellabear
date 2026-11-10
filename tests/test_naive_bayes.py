"""Tes Naive Bayes: `python -m pytest tests/test_naive_bayes.py` dari root repo."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from naive_bayes import NaiveBayes  # noqa: E402

rng = np.random.default_rng(42)
N = 600
y_all = rng.integers(0, 2, N)
df_all = pd.DataFrame({
    "num": rng.normal(y_all * 1.5, 1.0),
    "cat": np.where(rng.random(N) < 0.8, np.where(y_all == 1, "a", "b"), "c"),
    "few": rng.integers(0, 3, N),
})
X_tr, y_tr, X_te, y_te = df_all[:450], y_all[:450], df_all[450:], y_all[450:]


def test_hand_computed_probability():
    X = pd.DataFrame({"f": ["x", "x", "y", "y", "y", "x"]})
    y = np.array([1, 1, 0, 0, 1, 0])
    p = NaiveBayes(alpha=1.0).fit(X, y).predict_proba(pd.DataFrame({"f": ["x", "y"]}))
    # P(x|1)=(2+1)/(3+3), P(x|0)=(1+1)/(3+3), prior 0,5 -> P(1|x)=0,6 ; P(1|y)=0,4
    assert np.allclose(p, [0.6, 0.4])


def test_missing_features_are_skipped():
    m = NaiveBayes().fit(X_tr, y_tr)
    prior1 = np.exp(m.log_prior_[1])
    all_nan = pd.DataFrame({"num": [np.nan], "cat": [np.nan], "few": [np.nan]})
    assert np.allclose(m.predict_proba(all_nan), prior1 / (prior1 + np.exp(m.log_prior_[0])))
    only_cat = pd.DataFrame({"num": [np.nan], "cat": ["a"], "few": [np.nan]})
    single = NaiveBayes().fit(X_tr[["cat"]], y_tr).predict_proba(pd.DataFrame({"cat": ["a"]}))
    assert np.allclose(m.predict_proba(only_cat), single)


def test_unseen_category_and_out_of_range_numeric():
    m = NaiveBayes().fit(X_tr, y_tr)
    odd = pd.DataFrame({"num": [-1e9, 1e9], "cat": ["zzz", "zzz"], "few": [99, -5]})
    p = m.predict_proba(odd)
    assert p.shape == (2,) and np.isfinite(p).all() and ((p > 0) & (p < 1)).all()


def test_bins_come_from_train_only():
    m = NaiveBayes(n_bins=5).fit(X_tr, y_tr)
    edges = m.edges_["num"].copy()
    m.predict_proba(X_te)
    assert len(edges) == 4 and (m.edges_["num"] == edges).all()
    assert m.edges_["cat"] is None and m.edges_["few"] is None


def test_shape_range_and_learns_signal():
    m = NaiveBayes().fit(X_tr, y_tr)
    p = m.predict_proba(X_te)
    assert p.shape == (len(X_te),) and ((p >= 0) & (p <= 1)).all()
    assert (m.predict(X_te) == y_te).mean() > max(y_te.mean(), 1 - y_te.mean())


def test_threshold_and_inputs_not_mutated():
    before = X_tr.copy()
    m = NaiveBayes().fit(X_tr, y_tr)
    assert X_tr.equals(before)
    assert m.predict(X_te, threshold=1.0).sum() == 0
    assert m.predict(X_te, threshold=0.0).sum() == len(X_te)


def test_save_and_load_give_identical_predictions():
    m = NaiveBayes().fit(X_tr, y_tr)
    with tempfile.TemporaryDirectory() as d:
        m.save(Path(d) / "nb.pkl")
        assert (NaiveBayes.load(Path(d) / "nb.pkl").predict_proba(X_te) == m.predict_proba(X_te)).all()

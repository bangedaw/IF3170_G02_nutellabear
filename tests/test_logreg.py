"""Tes Logistic Regression: `python -m pytest tests/test_logreg.py` dari root repo."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logreg import LogisticRegression, log_loss, sigmoid  # noqa: E402

X, y = make_classification(n_samples=1000, n_features=10, n_informative=4, n_redundant=2, random_state=42)
X = StandardScaler().fit_transform(X)
N = len(y)
ALPHA = 0.01


def fit_scratch(penalty, **kw):
    return LogisticRegression(lr=0.5, n_iter=20000, tol=1e-12, penalty=penalty, alpha=ALPHA, **kw).fit(X, y)


def fit_sklearn(penalty, l1_ratio=0.5):
    """alpha = 1 / (C n). Gaya API sklearn >= 1.8: l1_ratio 0 = L2, 1 = L1, tanpa penalty= (deprecated)."""
    C = 1 / (ALPHA * N)
    if penalty == "none":
        return SkLogisticRegression(C=np.inf, max_iter=20000, tol=1e-10).fit(X, y)
    ratio = {"l2": 0.0, "l1": 1.0, "elasticnet": l1_ratio}[penalty]
    return SkLogisticRegression(C=C, l1_ratio=ratio, solver="saga", max_iter=20000, tol=1e-10).fit(X, y)


def test_sigmoid_stable_and_correct():
    with np.errstate(over="raise", invalid="raise", divide="raise"):  # underflow ke 0 itu wajar
        out = sigmoid(np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0]))
    assert np.allclose(out, [0.0, 1.9287498479639178e-22, 0.5, 1.0, 1.0])
    z = np.linspace(-10, 10, 41)
    assert np.allclose(sigmoid(z), 1 / (1 + np.exp(-z)))


def test_log_loss_matches_sklearn_and_survives_zero():
    p = np.clip(np.random.default_rng(0).random(N), 0.01, 0.99)
    assert np.isclose(log_loss(y, p), sk_log_loss(y, p))
    assert np.isfinite(log_loss(np.array([1, 0]), np.array([0.0, 1.0])))


@pytest.mark.parametrize("penalty", ["none", "l2", "l1", "elasticnet"])
def test_matches_sklearn(penalty):
    ours, ref = fit_scratch(penalty), fit_sklearn(penalty)
    assert ours.converged_
    assert np.allclose(ours.coef_, ref.coef_[0], atol=1e-3)
    assert np.isclose(ours.intercept_, ref.intercept_[0], atol=1e-3)
    assert np.allclose(ours.predict_proba(X), ref.predict_proba(X)[:, 1], atol=1e-3)


def test_l1_gives_exact_zeros_and_l2_does_not():
    l1, l2 = fit_scratch("l1"), fit_scratch("l2")
    assert (l1.coef_ == 0).sum() > 0
    assert (l2.coef_ == 0).sum() == 0
    zeros = [(LogisticRegression(lr=0.5, n_iter=5000, penalty="l1", alpha=a).fit(X, y).coef_ == 0).sum()
             for a in (0.001, 0.02, 0.2)]
    assert zeros[0] <= zeros[1] <= zeros[2] and zeros[2] > zeros[0]


def test_elasticnet_between_l1_and_l2():
    l1_only = LogisticRegression(lr=0.5, n_iter=5000, penalty="elasticnet", alpha=ALPHA, l1_ratio=1.0).fit(X, y)
    l1 = LogisticRegression(lr=0.5, n_iter=5000, penalty="l1", alpha=ALPHA).fit(X, y)
    l2_only = LogisticRegression(lr=0.5, n_iter=5000, penalty="elasticnet", alpha=ALPHA, l1_ratio=0.0).fit(X, y)
    l2 = LogisticRegression(lr=0.5, n_iter=5000, penalty="l2", alpha=ALPHA).fit(X, y)
    assert np.allclose(l1_only.coef_, l1.coef_)
    assert np.allclose(l2_only.coef_, l2.coef_)


def test_history_full_batch_is_monotone_and_consistent():
    m = LogisticRegression(lr=0.5, n_iter=300, penalty="l2", alpha=ALPHA).fit(X, y)
    h = m.history_
    assert h["epoch"][0] == 0 and np.allclose(h["params"][0], 0)
    assert h["epoch"][-1] == m.n_iter_
    assert h["params"].shape == (len(h["epoch"]), X.shape[1] + 1)
    assert np.all(np.diff(h["loss"]) <= 1e-12)
    assert np.allclose(h["params"][-1], np.r_[m.intercept_, m.coef_])


def test_record_every_still_records_last_epoch():
    m = LogisticRegression(lr=0.5, n_iter=25, tol=0, record_every=10).fit(X, y)
    assert list(m.history_["epoch"]) == [0, 10, 20, 25]


def test_minibatch_reaches_similar_accuracy():
    full = LogisticRegression(lr=0.3, n_iter=200).fit(X, y)
    mini = LogisticRegression(lr=0.3, n_iter=200, batch_size=64).fit(X, y)
    assert abs((full.predict(X) == y).mean() - (mini.predict(X) == y).mean()) < 0.02


def test_not_converged_flag():
    assert not LogisticRegression(lr=0.01, n_iter=3).fit(X, y).converged_


def test_predict_threshold_and_shapes():
    m = fit_scratch("l2")
    p = m.predict_proba(X)
    assert p.shape == (N,) and ((p >= 0) & (p <= 1)).all()
    assert (m.predict(X) == (p >= 0.5)).all()
    assert m.predict(X, threshold=0.0).all() and not m.predict(X, threshold=1.0).any()
    with pytest.raises(ValueError):
        m.predict(X, threshold=1.5)


def test_accepts_dataframe_and_does_not_mutate_input():
    df, y_copy = pd.DataFrame(X.copy()), y.copy()
    ref = df.copy()
    m = LogisticRegression(lr=0.5, n_iter=100).fit(df, y_copy)
    assert df.equals(ref) and (y_copy == y).all()
    assert np.allclose(m.predict_proba(df), m.predict_proba(X))


def test_save_load_identical_predictions():
    m = fit_scratch("elasticnet", l1_ratio=0.5)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "lr.pkl"
        m.save(path)
        loaded = LogisticRegression.load(path)
        assert np.array_equal(loaded.predict_proba(X), m.predict_proba(X))
        assert np.array_equal(loaded.predict(X), m.predict(X))
        bad = Path(d) / "other.pkl"
        import pickle
        bad.write_bytes(pickle.dumps({"bukan": "model"}))
        with pytest.raises(TypeError):
            LogisticRegression.load(bad)


def test_input_validation():
    with pytest.raises(ValueError):
        LogisticRegression(penalty="ridge").fit(X, y)
    with pytest.raises(ValueError):
        LogisticRegression(lr=0).fit(X, y)
    with pytest.raises(ValueError):
        LogisticRegression(l1_ratio=2).fit(X, y)
    with pytest.raises(ValueError):
        LogisticRegression().fit(X, y + 1)  # label bukan 0/1
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(ValueError):
        LogisticRegression().fit(Xn, y)
    with pytest.raises(RuntimeError):
        LogisticRegression().predict(X)
    m = LogisticRegression(n_iter=5).fit(X, y)
    with pytest.raises(ValueError):
        m.predict(X[:, :3])


def test_video_written_for_two_parameter_model(tmp_path):
    x1 = X[:, [0]]
    m = LogisticRegression(lr=0.5, n_iter=60, penalty="none", tol=0).fit(x1, y)
    out = m.plot_loss_contour_video(x1, y, tmp_path / "lr.mp4", max_frames=8, grid=20, fps=5)
    assert Path(out).exists() and Path(out).stat().st_size > 0
    with pytest.raises(ValueError):
        LogisticRegression(n_iter=5).fit(X, y).plot_loss_contour_video(X, y, tmp_path / "x.mp4")

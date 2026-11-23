import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logreg import LogisticRegression
from logreg_optim import OptimizedLogisticRegression, lipschitz_constant

rng = np.random.default_rng(7)
base = rng.normal(size=(2000, 1))
X = np.column_stack([base + 0.3 * rng.normal(size=(2000, 1)) for _ in range(8)] + [rng.normal(size=(2000, 2))])
X = (X - X.mean(0)) / X.std(0)
w_true = np.r_[np.ones(8) * 0.4, 1.0, -1.0]
y = (rng.random(2000) < 1 / (1 + np.exp(-(X @ w_true)))).astype(int)


def test_lipschitz_constant_matches_eigenvalue():
    Z = np.column_stack([np.ones(len(X)), X])
    exact = np.linalg.eigvalsh(Z.T @ Z).max() / (4 * len(X)) + 0.01
    assert np.isclose(lipschitz_constant(X, 0.01), exact, rtol=1e-6)


def test_fista_beats_gd_at_equal_budget_and_matches_at_convergence():
    step = 1 / lipschitz_constant(X, 0.01)
    gd = OptimizedLogisticRegression("gd", lr=step, n_iter=40, penalty="l2", alpha=0.01, tol=0).fit(X, y)
    fista = OptimizedLogisticRegression("fista", n_iter=40, penalty="l2", alpha=0.01, tol=0).fit(X, y)
    assert fista.history_["loss"][-1] < gd.history_["loss"][-1] - 1e-4
    gd_long = OptimizedLogisticRegression("gd", lr=step, n_iter=20000, penalty="l2", alpha=0.01, tol=1e-12).fit(X, y)
    fista_long = OptimizedLogisticRegression("fista", n_iter=3000, penalty="l2", alpha=0.01, tol=1e-12).fit(X, y)
    assert np.allclose(gd_long.coef_, fista_long.coef_, atol=1e-3)


def test_fista_l1_matches_parent_ista_solution_with_fewer_epochs():
    parent = LogisticRegression(lr=1 / lipschitz_constant(X), n_iter=20000, penalty="l1", alpha=0.2, tol=1e-13).fit(X, y)
    m = OptimizedLogisticRegression("fista", n_iter=3000, penalty="l1", alpha=0.2, tol=1e-13).fit(X, y)
    assert (m.coef_ == 0).sum() == (parent.coef_ == 0).sum() > 0
    assert np.allclose(m.coef_, parent.coef_, atol=1e-3) and m.n_iter_ < parent.n_iter_
    assert m.history_["params"].shape[1] == X.shape[1] + 1 and m.history_["epoch"][0] == 0


def test_adam_reaches_the_l2_optimum_and_rejects_l1():
    ref = OptimizedLogisticRegression("fista", n_iter=3000, penalty="l2", alpha=0.01, tol=1e-12).fit(X, y)
    adam = OptimizedLogisticRegression("adam", lr=0.01, n_iter=300, penalty="l2", alpha=0.01, tol=0,
                                       batch_size=None).fit(X, y)
    assert adam.history_["loss"][-1] - ref.history_["loss"][-1] < 1e-3
    with pytest.raises(ValueError):
        OptimizedLogisticRegression("adam", lr=0.01, penalty="l1").fit(X, y)


def test_gd_option_is_identical_to_parent_class():
    a = OptimizedLogisticRegression("gd", lr=0.1, n_iter=200, penalty="elasticnet", alpha=0.01).fit(X, y)
    b = LogisticRegression(lr=0.1, n_iter=200, penalty="elasticnet", alpha=0.01).fit(X, y)
    assert np.array_equal(a.coef_, b.coef_) and a.intercept_ == b.intercept_

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from svm import SVM, GaussianKernel
from svm_wss import SVMWSS2, select_second_order

rng = np.random.default_rng(3)
X = rng.normal(size=(500, 4))
y = ((X[:, 0] ** 2 + X[:, 1] - 0.5 + 0.4 * rng.normal(size=500)) > 0).astype(int)


def dual_objective(model, X, y):
    ys = np.where(y == 1, 1.0, -1.0)
    a = np.zeros(len(y))
    a[model.support_] = model.dual_coef_ * ys[model.support_]
    K = model.kernel.compute(X, X)
    v = a * ys
    return 0.5 * v @ K @ v - a.sum()


def test_second_order_choice_matches_brute_force():
    K = GaussianKernel(0.5).compute(X[:60], X[:60])
    score = rng.normal(size=60)
    low = rng.random(60) < 0.6
    i = int(np.argmax(score))
    j = select_second_order(score, low, i, K[i], K[i, i], np.diag(K))
    best, best_val = None, np.inf
    for t in range(60):
        if low[t] and score[t] < score[i]:
            a = max(K[i, i] + K[t, t] - 2 * K[i, t], 1e-12)
            val = -(score[i] - score[t]) ** 2 / a
            if val < best_val:
                best, best_val = t, val
    assert j == best


def test_wss2_reaches_same_optimum_in_fewer_iterations():
    first = SVM(GaussianKernel(0.5), C=10, max_iter=200_000).fit(X, y)
    second = SVMWSS2(GaussianKernel(0.5), C=10, max_iter=200_000).fit(X, y)
    assert first.converged_ and second.converged_ and second.kkt_gap_ < second.tol
    assert second.n_iter_ < first.n_iter_
    assert abs(dual_objective(first, X, y) - dual_objective(second, X, y)) < 1e-2 * abs(dual_objective(first, X, y))
    assert np.mean(first.predict(X) == second.predict(X)) > 0.99

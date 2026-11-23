import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cost_sensitive import cost_sensitive_cost, cross_fit_cost, fit_threshold, fit_threshold_proba, threshold_curve


def test_cost_matches_hand_count():
    y = np.array([1, 1, 0, 0, 0, 1])
    pred = np.array([1, 0, 1, 1, 0, 0])
    assert cost_sensitive_cost(y, pred) == 2 * 2 + 1 * 2
    assert cost_sensitive_cost(y, pred, cost_fp=1, cost_fn=1) == 4
    assert cost_sensitive_cost(y, y) == 0


def test_curve_matches_brute_force():
    rng = np.random.default_rng(0)
    proba = np.round(rng.random(300), 1)
    y = (rng.random(300) < proba).astype(int)
    thresholds, costs = threshold_curve(proba, y)
    for t, c in zip(thresholds, costs):
        assert c == cost_sensitive_cost(y, (proba >= t).astype(int))


def test_fitted_threshold_never_worse_than_half():
    rng = np.random.default_rng(1)
    proba = rng.random(500)
    y = (rng.random(500) < proba).astype(int)
    t = fit_threshold_proba(proba, y)
    assert cost_sensitive_cost(y, proba >= t) <= cost_sensitive_cost(y, proba >= 0.5)


def test_threshold_moves_up_when_fp_is_costlier():
    rng = np.random.default_rng(2)
    proba = rng.random(20000)
    y = (rng.random(20000) < proba).astype(int)
    equal = fit_threshold_proba(proba, y, cost_fp=1, cost_fn=1)
    costly_fp = fit_threshold_proba(proba, y, cost_fp=2, cost_fn=1)
    assert costly_fp > equal


def test_calibrated_scores_give_theoretical_threshold():
    rng = np.random.default_rng(3)
    proba = rng.random(400000)
    y = (rng.random(400000) < proba).astype(int)
    assert abs(fit_threshold_proba(proba, y) - 2 / 3) < 0.03
    assert abs(fit_threshold_proba(proba, y, cost_fp=1, cost_fn=1) - 0.5) < 0.03


def test_degenerate_inputs():
    assert np.isfinite(fit_threshold_proba(np.full(10, 0.4), np.array([0, 1] * 5)))
    t = fit_threshold_proba(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    assert 0.2 < t <= 0.8


def test_fit_threshold_wrapper_and_cross_fit():
    class Fixed:
        def predict_proba(self, X):
            return np.asarray(X, dtype=float)

    rng = np.random.default_rng(4)
    proba = rng.random(2000)
    y = (rng.random(2000) < proba).astype(int)
    assert fit_threshold(Fixed(), proba, y) == fit_threshold_proba(proba, y)
    assert cross_fit_cost(proba, y) == cross_fit_cost(proba, y)
    assert cross_fit_cost(proba, y) >= cost_sensitive_cost(y, proba >= fit_threshold_proba(proba, y))

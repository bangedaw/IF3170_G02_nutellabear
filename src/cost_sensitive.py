import numpy as np


def cost_sensitive_cost(y, y_pred, cost_fp=2.0, cost_fn=1.0):
    y = np.asarray(y).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    fp = int(((y == 0) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == 0)).sum())
    return cost_fp * fp + cost_fn * fn


def threshold_curve(proba, y, cost_fp=2.0, cost_fn=1.0):
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y).astype(int)
    values, inverse = np.unique(proba, return_inverse=True)
    positives = np.bincount(inverse, weights=y, minlength=len(values))
    negatives = np.bincount(inverse, weights=1 - y, minlength=len(values))
    thresholds = np.concatenate([[values[0]], (values[:-1] + values[1:]) / 2,
                                 [np.nextafter(values[-1], np.inf)]])
    missed = np.concatenate([[0.0], np.cumsum(positives)])
    false_alarms = negatives.sum() - np.concatenate([[0.0], np.cumsum(negatives)])
    return thresholds, cost_fp * false_alarms + cost_fn * missed


def fit_threshold_proba(proba, y, cost_fp=2.0, cost_fn=1.0):
    thresholds, costs = threshold_curve(proba, y, cost_fp, cost_fn)
    theory = cost_fp / (cost_fp + cost_fn)
    best = np.flatnonzero(costs == costs.min())
    return float(thresholds[best[np.argmin(np.abs(thresholds[best] - theory))]])


def fit_threshold(model, X_val, y_val, cost_fp=2.0, cost_fn=1.0):
    return fit_threshold_proba(model.predict_proba(X_val), y_val, cost_fp, cost_fn)


def cross_fit_cost(proba, y, cost_fp=2.0, cost_fn=1.0, seed=42):
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    half = np.zeros(len(y), dtype=int)
    for label in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == label))
        half[idx] = np.arange(len(idx)) % 2
    total = 0.0
    for k in (0, 1):
        fit, test = half != k, half == k
        t = fit_threshold_proba(proba[fit], y[fit], cost_fp, cost_fn)
        total += cost_sensitive_cost(y[test], (proba[test] >= t).astype(int), cost_fp, cost_fn)
    return total

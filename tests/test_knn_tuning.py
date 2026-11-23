import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from knn import KNN
from knn_tuning import (BASELINE, SPACE, WeightedKNN, column_mutual_information, column_scale, cyclic_month,
                        evaluate_fold, log_skewed, proba_from_neighbors, random_configs, summarize)

rng = np.random.default_rng(42)
N = 400
y = rng.integers(0, 2, N)
X = pd.DataFrame({
    "signal": y + rng.normal(0, 0.7, N),
    "noise": rng.normal(0, 1, N),
    "cat=a": (rng.random(N) < 0.5).astype(int),
    "cat=b": (rng.random(N) < 0.5).astype(int),
})
Xq = X.iloc[:60].reset_index(drop=True)


def schema_frame(n=500, seed=1):
    r = np.random.default_rng(seed)
    target = r.integers(0, 2, n)
    return pd.DataFrame({
        "id": [f"r{i}" for i in range(n)],
        "num_adults": r.integers(1, 4, n),
        "num_children": np.where(r.random(n) < 0.1, np.nan, r.integers(0, 3, n)),
        "weekend_nights": r.integers(0, 3, n),
        "weekday_nights": r.integers(0, 5, n),
        "meal_package": r.choice(["Breakfast", "Half Board", "None"], n),
        "needs_parking": r.integers(0, 2, n),
        "room_category": r.choice(["Type A", "Type B", "Type C"], n),
        "days_before_arrival": r.integers(0, 300, n) + target * 40,
        "arrival_year": r.choice([2024, 2025], n),
        "arrival_month": r.integers(1, 13, n),
        "arrival_day": r.integers(1, 29, n),
        "booking_channel": r.choice(["Online", "Offline Agent", "Corporate"], n),
        "is_returning_guest": r.integers(0, 2, n),
        "prior_cancellations": r.integers(0, 2, n),
        "prior_completed_bookings": r.integers(0, 3, n),
        "nightly_rate_idr": np.where(r.random(n) < 0.05, np.nan, r.normal(1.2e6, 3e5, n)),
        "special_requests": r.integers(0, 4, n),
        "is_canceled": target,
    })


def test_proba_from_neighbors_matches_knn_for_every_k_and_weighting():
    for weights in ("uniform", "distance"):
        big = KNN(k=41, metric="manhattan", p=1, weights=weights).fit(X, y)
        dist, idx = big.kneighbors(Xq)
        labels = big.y_[idx]
        for k in (1, 5, 15, 41):
            direct = KNN(k=k, metric="manhattan", p=1, weights=weights).fit(X, y).predict_proba(Xq)
            assert np.allclose(proba_from_neighbors(dist, labels, k, weights), direct)


def test_default_weighted_knn_equals_plain_knn():
    plain = KNN(k=15, metric="manhattan", p=1, weights="distance").fit(X, y).predict_proba(Xq)
    wrapped = WeightedKNN(k=15, metric="manhattan", p=1, weights="distance").fit(X, y).predict_proba(Xq)
    assert np.array_equal(plain, wrapped)


def test_column_scaling_is_weighted_minkowski():
    model = WeightedKNN(k=5, metric="manhattan", p=1, gamma=1.0, top_k=3).fit(X, y)
    dist, idx = model.kneighbors(Xq)
    cols = X.to_numpy(float)[:, model.keep_]
    expected = (np.abs(Xq.to_numpy(float)[:, model.keep_][:, None, :] - cols[None, :, :]) * model.scale_).sum(2)
    assert np.allclose(dist, np.sort(expected, axis=1)[:, :5])


def test_top_k_keeps_the_most_informative_columns():
    model = WeightedKNN(top_k=2).fit(X, y)
    assert len(model.keep_) == 2
    assert "signal" in list(X.columns[model.keep_])
    assert set(model.keep_) == set(np.argsort(-model.mi_)[:2])


def test_mutual_information_is_zero_for_constant_and_high_for_signal():
    mi = column_mutual_information(np.column_stack([np.ones(N), y]), y)
    assert mi[0] == 0
    assert abs(mi[1] - (-(0.5 * np.log2(0.5) * 2))) < 0.02


def test_onehot_factor_only_scales_columns_with_equal_sign():
    scale = column_scale(np.array([0.4, 0.2, 0.1]), ["a", "c=x", "c=y"], gamma=0, onehot_factor=0.5)
    assert np.allclose(scale, [1, 0.5, 0.5])


def test_cyclic_month_makes_december_as_close_to_january_as_february():
    out = cyclic_month(pd.DataFrame({"arrival_month": [12, 1, 2]}))
    pts = out[["month_sin", "month_cos"]].to_numpy()
    assert np.isclose(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2]))
    assert "arrival_month" not in out


def test_log_skewed_does_not_mutate_input_and_handles_nan():
    df = pd.DataFrame({"days_before_arrival": [0.0, 10.0, np.nan], "nightly_rate_idr": [1.0, 2.0, 3.0],
                       "nightly_rate_per_guest": [1.0, 2.0, 3.0]})
    before = df.copy()
    out = log_skewed(df)
    assert df.equals(before) and np.isnan(out["days_before_arrival"].iloc[2])
    assert np.isclose(out["days_before_arrival"].iloc[1], np.log1p(10.0))


def test_save_and_load_round_trip(tmp_path):
    model = WeightedKNN(k=7, gamma=1.0, top_k=3, onehot_factor=0.5).fit(X, y)
    model.save(tmp_path / "m.npz")
    loaded = WeightedKNN.load(tmp_path / "m.npz")
    assert np.array_equal(model.predict_proba(Xq), loaded.predict_proba(Xq))


def test_random_configs_are_reproducible_unique_and_start_with_baseline():
    a, b = random_configs(10, seed=3), random_configs(10, seed=3)
    assert a == b and a[0] == BASELINE
    assert len({tuple(sorted(c.items())) for c in a}) == len(a) == 11
    assert all(c["top_k"] in SPACE["top_k"] and c["gamma"] in SPACE["gamma"] for c in a[1:])


def test_evaluate_fold_runs_on_schema_and_summarize_ranks():
    df = schema_frame()
    config = {**BASELINE, "cyclic": True, "log": True, "gamma": 1.0, "top_k": 12, "onehot_factor": 0.5}
    rows = pd.DataFrame(evaluate_fold(df, config, fold=0, ks=(5, 15), k_folds=3))
    assert len(rows) == 4 and set(rows["weights"]) == {"uniform", "distance"}
    assert rows["accuracy"].between(0, 1).all()
    ranked = summarize(rows)
    assert ranked["accuracy"].is_monotonic_decreasing

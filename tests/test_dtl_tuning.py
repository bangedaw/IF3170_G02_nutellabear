import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decision_tree import DecisionTree, TreeNode
from dtl_tuning import (BASELINE, SPACE, PrunedDecisionTree, count_nodes, evaluate_fold, prune, prune_node,
                        random_configs, summarize)


def hand_tree():
    leaf_a = TreeNode(np.array([9.0, 1.0]), 10, 1)
    leaf_b = TreeNode(np.array([1.0, 9.0]), 10, 1)
    return TreeNode(np.array([10.0, 10.0]), 20, 0, feature=0, threshold=0.5, children={0: leaf_a, 1: leaf_b},
                    branch_probabilities={0: 0.5, 1: 0.5})


def noisy_data(n, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = ((X["f0"] + 0.5 * X["f1"]) > 0).astype(int).to_numpy()
    flip = rng.random(n) < 0.2
    return X, np.where(flip, 1 - y, y)


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


def test_prune_matches_hand_computed_threshold():
    total = 20.0
    assert not prune_node(hand_tree(), 0.39, total)[1].is_leaf()
    assert prune_node(hand_tree(), 0.40, total)[1].is_leaf()
    assert prune_node(hand_tree(), 0.41, total)[1].is_leaf()


def test_prune_does_not_modify_original_and_zero_alpha_is_identity():
    X, y = noisy_data(800, 0)
    tree = DecisionTree().fit(X, y)
    nodes = count_nodes(tree.root)
    assert prune(tree, 0) is tree and prune(tree, None) is tree
    smaller = prune(tree, 5e-3)
    assert count_nodes(tree.root) == nodes and count_nodes(smaller.root) < nodes


def test_node_count_is_non_increasing_in_alpha_and_huge_alpha_gives_one_leaf():
    X, y = noisy_data(800, 0)
    tree = DecisionTree().fit(X, y)
    counts = [count_nodes(prune(tree, a).root) for a in (0, 1e-4, 1e-3, 1e-2, 1.0)]
    assert counts == sorted(counts, reverse=True) and counts[-1] == 1
    assert len(set(prune(tree, 1.0).predict_proba(X))) == 1


def test_pruning_helps_on_noisy_labels():
    X, y = noisy_data(2000, 1)
    X_te, y_te = noisy_data(2000, 2)
    tree = DecisionTree(max_depth=None).fit(X, y)
    accuracy = lambda m: ((m.predict_proba(X_te) >= 0.5).astype(int) == y_te).mean()
    assert accuracy(prune(tree, 5e-4)) > accuracy(tree) + 0.03
    assert count_nodes(prune(tree, 5e-4).root) < count_nodes(tree.root) / 2


def test_wrapper_keeps_informative_columns_and_handles_missing_and_text():
    r = np.random.default_rng(3)
    n = 1200
    y = r.integers(0, 2, n)
    X = pd.DataFrame({"signal": y + r.normal(0, 0.5, n), "noise": r.normal(size=n), "other": r.normal(size=n),
                      "meal_package": r.choice(["a", "b", "c"], n)})
    X.loc[X.index[:60], "signal"] = np.nan
    X.loc[X.index[60:120], "meal_package"] = np.nan
    model = PrunedDecisionTree(top_k=2, alpha=1e-4).fit(X, y)
    assert "signal" in model.columns_ and len(model.columns_) == 2
    proba = model.predict_proba(X)
    assert proba.shape == (n,) and ((proba >= 0) & (proba <= 1)).all()
    assert np.array_equal(model.predict(X), (proba >= 0.5).astype(int))
    assert model.n_nodes_ <= count_nodes(model.full_tree_.root)


def test_wrapper_is_leak_free_fit_uses_only_given_rows():
    X, y = noisy_data(600, 4)
    a = PrunedDecisionTree(top_k=3, alpha=1e-3).fit(X.iloc[:400], y[:400])
    b = PrunedDecisionTree(top_k=3, alpha=1e-3).fit(X.iloc[:400], y[:400])
    assert np.array_equal(a.predict_proba(X.iloc[400:]), b.predict_proba(X.iloc[400:]))


def test_save_and_load_round_trip(tmp_path):
    X, y = noisy_data(600, 5)
    model = PrunedDecisionTree(top_k=3, alpha=1e-3).fit(X, y)
    model.save(tmp_path / "m.bin")
    loaded = PrunedDecisionTree.load(tmp_path / "m.bin")
    assert np.array_equal(model.predict_proba(X), loaded.predict_proba(X)) and loaded.columns_ == model.columns_


def test_random_configs_are_reproducible_unique_and_start_with_baseline():
    a, b = random_configs(12, seed=3), random_configs(12, seed=3)
    assert a == b and a[0] == BASELINE
    assert len({tuple(sorted(c.items())) for c in a}) == 13
    assert all(c["top_k"] in SPACE["top_k"] and c["max_depth"] in SPACE["max_depth"] for c in a)


def test_evaluate_fold_and_summarize_on_schema_frame():
    df = schema_frame()
    rows = pd.DataFrame(evaluate_fold(df, {**BASELINE, "derived": False, "top_k": 12, "min_gain": 1e-4}, 0,
                                      alphas=(0, 1e-4, 1.0), k_folds=3))
    assert len(rows) == 3 and rows["nodes"].is_monotonic_decreasing and rows["nodes"].iloc[-1] == 1
    assert rows["accuracy"].between(0, 1).all()
    assert summarize(rows)["accuracy"].is_monotonic_decreasing

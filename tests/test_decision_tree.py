"""Jalankan dari root: python -m pytest tests/test_decision_tree.py -q."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.decision_tree import DecisionTree, entropy, gain_ratio, information_gain


def walk(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children.values())


def test_entropy_and_gain_hand_calculation():
    assert entropy([]) == 0
    assert entropy([0, 0]) == 0
    assert entropy([0, 1]) == 1
    expected = -0.75 * np.log2(0.75) - 0.25 * np.log2(0.25)
    assert entropy([0, 1], [3, 1]) == pytest.approx(expected)
    y = [0, 0, 1, 1]
    assert information_gain(y, [0, 0, 1, 1]) == 1
    assert gain_ratio(y, [0, 0, 1, 1]) == 1
    assert information_gain(y, [0, 1, 0, 1]) == 0
    assert gain_ratio(y, [0, 0, 0, 0]) == 0


def test_missing_gain_uses_known_parent_and_penalty():
    # Known: pure branches, 1 bit gain. Missing contributes 1/3 of all weight.
    y, branches = [0, 1, 1], np.array(["a", "b", None], dtype=object)
    assert information_gain(y, branches) == pytest.approx(2 / 3)
    assert gain_ratio(y, branches) == pytest.approx((2 / 3) / np.log2(3))
    assert gain_ratio(y, [np.nan] * 3) == 0


def test_numeric_search_matches_independent_exhaustive_partitions():
    rng = np.random.default_rng(42)
    for _ in range(20):
        x = rng.integers(0, 12, 50).astype(float)
        x[rng.random(50) < 0.2] = np.nan
        y = rng.integers(0, 2, 50)
        weights = rng.uniform(0.5, 2, 50)
        model = DecisionTree(min_samples_leaf=0.01)
        values = np.unique(x[~np.isnan(x)])
        thresholds = (values[:-1] + values[1:]) / 2
        scores = []
        for threshold in thresholds:
            branches = np.where(np.isnan(x), np.nan, (x > threshold).astype(float))
            scores.append(gain_ratio(y, branches, weights))
        split = model._best_numeric(x, y, weights, 0)
        if max(scores) <= 1e-12:
            assert split is None
        else:
            assert split.threshold == thresholds[np.argmax(scores)]
            assert split.ratio == pytest.approx(max(scores))


def test_gain_filter_rejects_high_ratio_low_gain_attribute():
    # First attribute separates a rare pure group: large ratio, small absolute gain.
    y = np.array([0] * 10 + [1] * 10)
    rare = np.array([0] + [1] * 19)
    useful = np.array([0] * 7 + [1] * 3 + [0] * 3 + [1] * 7)
    assert gain_ratio(y, rare) > gain_ratio(y, useful)
    assert information_gain(y, rare) < information_gain(y, useful)
    model = DecisionTree(categorical_features=[0, 1], max_depth=1)
    model.fit(np.column_stack((rare, useful)), y)
    assert model.root.feature == 1


def test_numeric_reuse_learns_interval():
    X = np.arange(6, dtype=float)[:, None]
    y = np.array([0, 0, 1, 1, 0, 0])
    model = DecisionTree().fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)
    assert model.depth_ == 2
    assert all(n.feature == 0 for n in walk(model.root) if not n.is_leaf())


def test_native_categories_have_no_ordinal_order():
    X = np.array([[0], [0], [1], [1], [2], [2]], dtype=float)
    y = [1, 1, 0, 0, 1, 1]
    model = DecisionTree(categorical_features=[0]).fit(X, y)
    assert model.root.categorical
    assert len(model.root.children) == 3
    assert model.depth_ == 1
    np.testing.assert_array_equal(model.predict(X), y)
    assert model.predict_proba([[99]])[0] == pytest.approx(2 / 3)


def test_fractional_training_and_unknown_predictions_by_hand():
    X = np.array([["a"], ["a"], ["b"], [None]], dtype=object)
    model = DecisionTree(categorical_features=[0], min_samples_leaf=0.1).fit(X, [0, 0, 1, 1])
    # Known branch priors: a=2/3, b=1/3. Missing positive divides accordingly.
    a, b = model.root.children["a"], model.root.children["b"]
    np.testing.assert_allclose(a.class_counts, [2, 2 / 3])
    np.testing.assert_allclose(b.class_counts, [0, 4 / 3])
    np.testing.assert_allclose(model.predict_proba([["a"], ["b"], [None], ["new"]]),
                               [0.25, 1, 0.5, 0.5])


def test_weight_conservation_and_prediction_mixture_across_levels():
    X = np.array([["a", 0], ["a", 1], ["a", 2], ["b", 0],
                  ["b", 1], ["b", 2], [None, 2], ["b", np.nan]], dtype=object)
    model = DecisionTree(categorical_features=[0], min_samples_leaf=0.1).fit(
        X, [0, 0, 1, 1, 1, 0, 1, 0])
    assert model.depth_ >= 2
    for node in walk(model.root):
        if not node.is_leaf():
            np.testing.assert_allclose(sum(c.class_counts for c in node.children.values()),
                                       node.class_counts)
            assert sum(node.branch_probabilities.values()) == pytest.approx(1)
    # If every feature is unknown, mixing leaves preserves the training prior.
    assert model.predict_proba([[None, np.nan]])[0] == pytest.approx(0.5)


def test_numeric_missing_by_hand():
    model = DecisionTree(min_samples_leaf=0.1).fit([[0], [0], [2], [np.nan]], [0, 0, 1, 1])
    assert model.root.threshold == 1
    np.testing.assert_allclose(model.predict_proba([[0], [2], [np.nan]]), [0.25, 1, 0.5])


def test_batch_prediction_matches_independent_recursive_traversal():
    rng = np.random.default_rng(42)
    X = rng.integers(0, 4, (150, 3)).astype(float)
    y = ((X[:, 0] == 1) ^ (X[:, 1] > 1) ^ (X[:, 2] == 2)).astype(int)
    X[rng.random(X.shape) < 0.15] = np.nan
    original = X.copy()
    model = DecisionTree(categorical_features=[0], max_depth=5, min_samples_leaf=0.2).fit(X, y)
    np.testing.assert_array_equal(X, original)
    queries = np.vstack((X, [99, np.nan, 2], [np.nan] * 3))

    def reference(node, row):
        if node.is_leaf():
            return node.probability
        value = row[node.feature]
        key = value if node.categorical else int(value > node.threshold)
        if not np.isnan(value) and key in node.children:
            return reference(node.children[key], row)
        return sum(node.branch_probabilities[k] * reference(child, row)
                   for k, child in node.children.items())

    expected = [reference(model.root, row) for row in queries]
    np.testing.assert_allclose(model.predict_proba(queries), expected, atol=1e-12)


def test_identical_features_break_ties_by_column_order():
    model = DecisionTree().fit([[0, 0], [1, 1], [2, 2], [3, 3]], [0, 0, 1, 1])
    assert model.root.feature == 0


def test_no_depth_limit_and_refitting():
    model = DecisionTree(max_depth=None).fit([[0], [1]], [0, 1])
    model.fit([[0, 0], [1, 1]], [1, 1])
    assert model.n_features_in_ == 2 and model.n_nodes_ == 1
    np.testing.assert_array_equal(model.predict([[99, 99]]), [1])


@pytest.mark.parametrize("X,y", [([[1], [1]], [0, 1]),
                                 ([[np.nan], [np.nan]], [0, 1]),
                                 ([[0], [1]], [1, 1])])
def test_terminal_cases(X, y):
    model = DecisionTree().fit(X, y)
    assert model.root.is_leaf()
    np.testing.assert_allclose(model.predict_proba(X), np.mean(y))


def test_stopping_rules_and_rare_categories():
    X, y = np.arange(6)[:, None], [0, 0, 0, 1, 1, 1]
    for kwargs in [dict(max_depth=0), dict(min_samples_split=7),
                   dict(min_samples_leaf=4), dict(min_gain=1)]:
        assert DecisionTree(**kwargs).fit(X, y).root.is_leaf()
    model = DecisionTree(categorical_features=[0], min_samples_leaf=2)
    assert model.fit([["a"], ["a"], ["b"]], [0, 0, 1]).root.is_leaf()


def test_interface_save_load_determinism_and_input_unchanged(tmp_path):
    X = np.array([[0, "a"], [1, "a"], [2, "b"], [None, "b"]], dtype=object)
    y = np.array([0, 0, 1, 1])
    original = X.copy()
    model = DecisionTree(categorical_features=[1], min_samples_leaf=0.1)
    assert model.fit(X, y) is model
    assert X.tolist() == original.tolist()
    np.testing.assert_array_equal(y, [0, 0, 1, 1])
    p = model.predict_proba(X)
    assert p.shape == (4,) and np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    np.testing.assert_array_equal(model.predict(X, threshold=0.7), p >= 0.7)
    assert model.predict_proba(np.empty((0, 2), dtype=object)).shape == (0,)
    path = tmp_path / "dtl.pkl"
    model.save(path)
    loaded = DecisionTree.load(path)
    np.testing.assert_array_equal(loaded.predict_proba(X), p)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
    again = DecisionTree(categorical_features=[1], min_samples_leaf=0.1).fit(X, y)
    np.testing.assert_array_equal(again.predict_proba(X), p)
    assert again.root.feature == model.root.feature
    assert again.root.threshold == model.root.threshold


def test_midpoint_extreme_and_adjacent_floats():
    for a, b in [(1e308, 1.1e308), (1.0, np.nextafter(1.0, np.inf))]:
        model = DecisionTree().fit([[a], [b]], [0, 1])
        assert np.isfinite(model.root.threshold)
        assert a <= model.root.threshold < b
        np.testing.assert_array_equal(model.predict([[a], [b]]), [0, 1])


@pytest.mark.parametrize("kwargs", [dict(max_depth=-1), dict(max_depth=1.2),
    dict(min_samples_split=1), dict(min_samples_split=True),
    dict(min_samples_leaf=0), dict(min_samples_leaf=np.nan), dict(min_gain=-1),
    dict(categorical_features=[1]), dict(categorical_features=[0, 0])])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        DecisionTree(**kwargs).fit([[0], [1]], [0, 1])


@pytest.mark.parametrize("X,y", [([], []), ([[0]], []), ([[0]], [2]),
    ([[0]], [[0]]), ([[np.inf]], [0]), ([["bad"]], [0]), ([[1 + 2j]], [0])])
def test_invalid_training_data(X, y):
    with pytest.raises((ValueError, TypeError)):
        DecisionTree().fit(X, y)


def test_fitted_and_prediction_validation(tmp_path):
    with pytest.raises(RuntimeError):
        DecisionTree().predict([[0]])
    with pytest.raises(RuntimeError):
        DecisionTree().save(tmp_path / "unfitted.pkl")
    model = DecisionTree().fit([[0], [1]], [0, 1])
    for X in [[[0, 1]], [0], [[np.inf]]]:
        with pytest.raises(ValueError):
            model.predict(X)
    for threshold in [-0.1, 1.1, np.nan, "bad"]:
        with pytest.raises(ValueError):
            model.predict([[0]], threshold=threshold)


@pytest.mark.parametrize("weights", [[1], [-1, 1], [np.nan, 1], [np.inf, 1]])
def test_invalid_entropy_weights(weights):
    with pytest.raises(ValueError):
        entropy([0, 1], weights)


def test_invalid_branch_shape_and_mixed_categories():
    for score in (information_gain, gain_ratio):
        with pytest.raises(ValueError):
            score([0, 1], [0])
    with pytest.raises(ValueError):
        DecisionTree(categorical_features=[0]).fit(np.array([["a"], [1]], dtype=object), [0, 1])


def test_simple_numeric_case_matches_sklearn_when_available():
    tree = pytest.importorskip("sklearn.tree", reason="sklearn tidak terpasang")
    X, y = np.arange(8)[:, None], [0] * 4 + [1] * 4
    model = DecisionTree().fit(X, y)
    reference = tree.DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.001,
                                           random_state=42).fit(X, y)
    assert model.root.threshold == reference.tree_.threshold[0]
    np.testing.assert_array_equal(model.predict(X), reference.predict(X))

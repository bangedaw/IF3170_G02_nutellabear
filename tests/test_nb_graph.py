import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt
from naive_bayes import NaiveBayes
from nb_graph import feature_class_mi, plot_nb_graph

rng = np.random.default_rng(42)
N = 6000
y = rng.integers(0, 2, N)
X = pd.DataFrame({
    "informative": np.where(y == 1, "a", "b"),
    "noise": rng.choice(["u", "v", "w"], N),
    "number": rng.normal(y * 1.0, 1.0, N),
})


def smoothed_mi(feature, target, alpha=1.0):
    levels = sorted(set(feature))
    counts = np.array([[np.sum((feature == level) & (target == c)) for level in levels] for c in (0, 1)], dtype=float)
    counts = np.hstack([counts, np.zeros((2, 1))])
    conditional = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * counts.shape[1])
    prior = (np.bincount(target, minlength=2) + alpha) / (len(target) + 2 * alpha)
    marginal = prior @ conditional
    return float((prior[:, None] * conditional * np.log2(conditional / marginal)).sum())


def test_mi_ranks_informative_feature_first():
    mi = feature_class_mi(NaiveBayes().fit(X, y))
    assert mi.index[0] == "informative"
    assert mi["informative"] > 0.99
    assert mi["noise"] < 0.005
    assert (mi >= 0).all()


def test_mi_matches_count_based_estimate_with_same_smoothing():
    mi = feature_class_mi(NaiveBayes().fit(X, y))
    for column in ("informative", "noise"):
        assert abs(mi[column] - smoothed_mi(X[column].to_numpy(), y)) < 1e-9


def test_plot_writes_image_and_handles_missing_values(tmp_path):
    data = X.copy()
    data.loc[data.index[:100], "number"] = np.nan
    data.loc[data.index[100:150], "noise"] = np.nan
    path = tmp_path / "graph.png"
    fig, mi = plot_nb_graph(NaiveBayes().fit(data, y), path)
    plt.close(fig)
    assert path.exists() and path.stat().st_size > 5000
    assert list(mi.index) == list(mi.sort_values(ascending=False).index)
    assert set(mi.index) == set(X.columns)

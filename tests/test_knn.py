"""Tes KNN: `python -m pytest tests/test_knn.py` dari root repo."""
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from knn import KNN  # noqa: E402

X, y = make_classification(n_samples=800, n_features=6, random_state=42)
X_tr, y_tr, X_te = X[:600], y[:600], X[600:]


def test_distances_match_sklearn():
    for metric, p in [("euclidean", 2), ("manhattan", 1), ("minkowski", 3)]:
        m = KNN(metric=metric, p=p).fit(X_tr, y_tr)
        ref = pairwise_distances(X_te, X_tr, metric=metric, **({"p": p} if metric == "minkowski" else {}))
        assert np.allclose(m._pairwise_distances(X_te), ref)


def test_proba_matches_sklearn():
    for metric, p in [("euclidean", 2), ("manhattan", 1), ("minkowski", 3)]:
        for weights in ["uniform", "distance"]:
            m = KNN(7, metric, p, weights, batch_size=50).fit(X_tr, y_tr)  # batch kecil: uji jalur batching
            ref = KNeighborsClassifier(7, weights=weights, metric=metric, p=p, algorithm="brute").fit(X_tr, y_tr)
            assert np.allclose(m.predict_proba(X_te), ref.predict_proba(X_te)[:, 1])


def test_zero_distance_and_save_load():
    m = KNN(3, weights="distance").fit(X_tr, y_tr)
    assert (m.predict(X_tr) == y_tr).all()  # titik latih sendiri berjarak 0 dan menentukan labelnya
    with tempfile.TemporaryDirectory() as d:
        m.save(Path(d) / "knn.pkl")
        assert (KNN.load(Path(d) / "knn.pkl").predict_proba(X_te) == m.predict_proba(X_te)).all()

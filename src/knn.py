"""KNN from scratch (NumPy).

Referensi: E. Ferdian, slide "KNN", IF3170 2026; P. Cunningham & S. J. Delany, "k-Nearest Neighbour Classifiers:
2nd Edition", arXiv:2004.04523, 2020. Penanganan jarak 0 pada weights="distance" mengikuti scikit-learn.
"""
import pickle

import numpy as np


class KNN:
    """metric: "euclidean", "manhattan", atau "minkowski" (pakai p). weights: "uniform" atau "distance"."""

    def __init__(self, k=5, metric="euclidean", p=2, weights="uniform", batch_size=256):
        self.k, self.metric, self.p, self.weights, self.batch_size = k, metric, p, weights, batch_size

    def fit(self, X, y):
        self.X_ = np.array(X, dtype=float)  # salinan, X pemanggil tidak diubah
        self.y_ = np.array(y).astype(int)
        return self

    def _pairwise_distances(self, Xq):
        """Matriks jarak (len(Xq), n_train)."""
        p = {"euclidean": 2, "manhattan": 1}.get(self.metric, self.p)
        if p == 2:  # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab, cukup satu perkalian matriks
            sq = (Xq ** 2).sum(1)[:, None] + (self.X_ ** 2).sum(1)[None, :] - 2 * Xq @ self.X_.T
            return np.sqrt(np.maximum(sq, 0))
        D = np.zeros((len(Xq), len(self.X_)))
        for j in range(Xq.shape[1]):  # per fitur, agar tidak membentuk array 3D (batch, n_train, d)
            D += np.abs(Xq[:, j, None] - self.X_[None, :, j]) ** p
        return D if p == 1 else D ** (1 / p)

    def kneighbors(self, X):
        """(jarak, indeks) k tetangga terdekat, terurut dari yang terdekat. Diproses per batch."""
        X = np.asarray(X, dtype=float)
        dist, idx = [], []
        for s in range(0, len(X), self.batch_size):
            D = self._pairwise_distances(X[s:s + self.batch_size])
            nn = np.argpartition(D, self.k - 1, axis=1)[:, :self.k]  # k terkecil, belum terurut
            d = np.take_along_axis(D, nn, 1)
            order = np.argsort(d, axis=1, kind="stable")
            dist.append(np.take_along_axis(d, order, 1))
            idx.append(np.take_along_axis(nn, order, 1))
        return np.vstack(dist), np.vstack(idx)

    def predict_proba(self, X):
        """Proporsi kelas 1 di antara k tetangga (berbobot 1/jarak bila weights="distance")."""
        dist, idx = self.kneighbors(X)
        labels = self.y_[idx]
        if self.weights == "uniform":
            return labels.mean(1)
        with np.errstate(divide="ignore"):
            w = 1 / dist
        exact = np.isinf(w)
        w[exact.any(1)] = exact[exact.any(1)]  # ada jarak 0: hanya titik identik yang dihitung
        return (w * labels).sum(1) / w.sum(1)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

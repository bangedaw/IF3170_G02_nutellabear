"""Naive Bayes from scratch (NumPy): fitur kategorikal dan numerik berbin kuantil, smoothing Laplace.

Referensi: N. U. Maulidevi, slide "Bayes Net", IF3170 2024.
"""
import pickle

import numpy as np
import pandas as pd


class NaiveBayes:
    """alpha: smoothing Laplace. n_bins: jumlah bin kuantil untuk fitur numerik bernilai unik banyak.
    max_categorical: fitur numerik dengan nilai unik <= ini diperlakukan sebagai kategori."""

    def __init__(self, alpha=1.0, n_bins=20, max_categorical=10):
        self.alpha, self.n_bins, self.max_categorical = alpha, n_bins, max_categorical

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y).astype(int)
        self.columns_ = list(X.columns)
        self.edges_, self.levels_, self.log_lik_ = {}, {}, {}

        n_class = np.bincount(y, minlength=2)
        self.log_prior_ = np.log((n_class + self.alpha) / (len(y) + 2 * self.alpha))

        for c in self.columns_:
            col = X[c]
            if pd.api.types.is_numeric_dtype(col) and col.nunique() > self.max_categorical:
                qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
                self.edges_[c] = np.unique(np.quantile(col.dropna().to_numpy(dtype=float), qs))
                self.levels_[c] = None
                n_levels = len(self.edges_[c]) + 1
            else:
                self.edges_[c] = None
                self.levels_[c] = sorted(col.dropna().unique())
                n_levels = len(self.levels_[c])

            codes = self._codes(c, col)
            counts = np.zeros((2, n_levels + 1))
            for k in (0, 1):
                seen = codes[(y == k) & (codes >= 0)]
                counts[k] = np.bincount(seen, minlength=n_levels + 1)
            self.log_lik_[c] = np.log((counts + self.alpha) / (counts.sum(1, keepdims=True) + self.alpha * (n_levels + 1)))
        return self

    def _codes(self, c, col):
        if self.edges_[c] is not None:
            v = col.to_numpy(dtype=float)
            codes = np.searchsorted(self.edges_[c], v, side="right")
            return np.where(np.isnan(v), -1, codes)
        codes = pd.Index(self.levels_[c]).get_indexer(col).astype(int)
        unseen = (codes == -1) & col.notna().to_numpy()
        return np.where(unseen, len(self.levels_[c]), codes)

    def predict_proba(self, X):
        X = pd.DataFrame(X)
        score = np.tile(self.log_prior_, (len(X), 1))
        for c in self.columns_:
            codes = self._codes(c, X[c])
            known = codes >= 0
            score[known] += self.log_lik_[c][:, codes[known]].T
        peak = score.max(axis=1, keepdims=True)
        log_norm = peak + np.log(np.exp(score - peak).sum(axis=1, keepdims=True))
        return np.exp(score - log_norm)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

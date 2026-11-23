"""Naive Bayes from scratch (NumPy): fitur kategorikal dan numerik berbin, smoothing Laplace.

Referensi: N. U. Maulidevi, slide "Bayes Net", IF3170 2024.
Diskretisasi MDL (bonus 6): U. M. Fayyad, K. B. Irani, "Multi-Interval Discretization of Continuous-Valued Attributes
for Classification Learning", Proc. IJCAI-93, hlm. 1022-1028, https://ijcai.org/proceedings/1993/141.
"""
import pickle

import numpy as np
import pandas as pd


def _entropy(counts):
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, n, out=np.zeros_like(counts), where=n > 0)
    logp = np.zeros_like(p)
    np.log2(p, out=logp, where=p > 0)
    return -(p * logp).sum(axis=-1)


def mdl_cut_points(x, y):
    """Fayyad & Irani (1993): bagi rekursif pada ambang dengan entropi kelas berbobot terkecil. Potongan diterima bila
    Gain > (log2(N - 1) + Delta) / N, dengan Delta = log2(3^k - 2) - (k Ent(S) - k1 Ent(S1) - k2 Ent(S2)).
    Ambang kandidat adalah titik tengah antarnilai berbeda; nilai hilang diabaikan."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).astype(int)
    known = ~np.isnan(x)
    values, inverse = np.unique(x[known], return_inverse=True)
    counts = np.zeros((len(values), 2))
    np.add.at(counts, (inverse, y[known]), 1)
    cuts, stack = [], [(0, len(values))]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        block = counts[lo:hi]
        total = block.sum(axis=0)
        n = total.sum()
        left = np.cumsum(block, axis=0)[:-1]
        right = total - left
        weighted = (left.sum(1) * _entropy(left) + right.sum(1) * _entropy(right)) / n
        b = int(np.argmin(weighted))
        ent_s = float(_entropy(total))
        k, k1, k2 = (int((c > 0).sum()) for c in (total, left[b], right[b]))
        delta = np.log2(3 ** k - 2) - (k * ent_s - k1 * _entropy(left[b]) - k2 * _entropy(right[b]))
        if ent_s - weighted[b] > (np.log2(n - 1) + delta) / n:
            cuts.append((values[lo + b] + values[lo + b + 1]) / 2)
            stack += [(lo, lo + b + 1), (lo + b + 1, hi)]
    return np.sort(np.array(cuts, dtype=float))


class NaiveBayes:
    """alpha: smoothing Laplace. n_bins: jumlah bin kuantil untuk fitur numerik bernilai unik banyak.
    max_categorical: fitur numerik dengan nilai unik <= ini diperlakukan sebagai kategori.
    discretization: "quantile" (n_bins interval kuantil) atau "mdl" (ambang Fayyad & Irani dari label data latih)."""

    def __init__(self, alpha=1.0, n_bins=20, max_categorical=10, discretization="quantile"):
        self.alpha, self.n_bins, self.max_categorical = alpha, n_bins, max_categorical
        self.discretization = discretization

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
                if self.discretization == "mdl":
                    self.edges_[c] = mdl_cut_points(col.to_numpy(dtype=float), y)
                else:
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

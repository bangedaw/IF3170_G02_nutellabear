"""SVM from scratch (NumPy) dengan SMO.

Referensi:
- Platt, J. (1998). Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines. MSR-TR-98-14.
- Keerthi, S. S., dkk. (2001). Improvements to Platt's SMO Algorithm for SVM Classifier Design. Neural Computation 13(3).
  (pemilihan pasangan "maximal violating pair" yang dipakai di sini)
- Platt, J. (1999). Probabilistic Outputs for Support Vector Machines. (Platt scaling)
- Lin, H.-T., Lin, C.-J., Weng, R. C. (2007). A Note on Platt's Probabilistic Outputs for Support Vector Machines.
  Machine Learning 68(3). (target ter-smoothing untuk Platt scaling)
"""
import pickle

import numpy as np


class LinearKernel:
    def compute(self, A, B):
        return A @ B.T


class PolynomialKernel:
    def __init__(self, degree=3, gamma=None, coef0=1.0):
        self.degree, self.gamma, self.coef0 = degree, gamma, coef0

    def compute(self, A, B):
        gamma = self.gamma if self.gamma is not None else 1.0 / A.shape[1]
        return (gamma * (A @ B.T) + self.coef0) ** self.degree


class GaussianKernel:
    def __init__(self, gamma=None):
        self.gamma = gamma

    def compute(self, A, B):
        gamma = self.gamma if self.gamma is not None else 1.0 / A.shape[1]
        sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * (A @ B.T)
        return np.exp(-gamma * np.maximum(sq, 0))


def _stratified_subsample(y, size, rng):
    """Indeks subsampel berstratifikasi sebanyak `size`, sisanya dikembalikan terpisah."""
    picked = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        picked.append(rng.choice(idx, round(size * len(idx) / len(y)), replace=False))
    picked = np.sort(np.concatenate(picked))
    return picked, np.setdiff1d(np.arange(len(y)), picked)


def _fit_platt(f, y, n_iter=100):
    """Cari A, B pada p = 1 / (1 + exp(A f + B)) dengan Newton (Lin dkk., 2007)."""
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    t = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    A, B = 0.0, np.log((n_neg + 1) / (n_pos + 1))

    def loss(A, B):
        z = A * f + B  # log-loss stabil: t*z + log(1 + exp(-z))
        return np.sum(t * z + np.logaddexp(0, -z))

    for _ in range(n_iter):
        p = 1 / (1 + np.exp(np.clip(A * f + B, -500, 500)))
        g = np.array([np.dot(f, t - p), np.sum(t - p)])
        w = p * (1 - p)
        H = np.array([[np.dot(f * f, w), np.dot(f, w)], [np.dot(f, w), w.sum()]]) + 1e-12 * np.eye(2)
        step = np.linalg.solve(H, g)
        old, s = loss(A, B), 1.0
        while s > 1e-8 and loss(A - s * step[0], B - s * step[1]) > old:
            s /= 2
        A, B = A - s * step[0], B - s * step[1]
        if np.abs(g).max() < 1e-5:
            break
    return A, B


class SVM:
    """SVM klasifikasi biner, dual diselesaikan dengan SMO.

    Strategi skala data: matriks kernel O(n^2) untuk 27 ribu baris tidak muat, jadi bila n > max_samples,
    SVM dilatih pada subsampel berstratifikasi sebanyak max_samples (matriks kernel penuh di-cache).
    Baris sisanya dipakai untuk Platt scaling, sehingga kalibrasi tidak dilakukan pada data latih SVM.
    """

    def __init__(self, kernel=None, C=1.0, tol=1e-3, max_iter=100_000, max_samples=4000,
                 n_calib=5000, random_state=42):
        self.kernel = kernel if kernel is not None else GaussianKernel()
        self.C, self.tol, self.max_iter = C, tol, max_iter
        self.max_samples, self.n_calib, self.random_state = max_samples, n_calib, random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y01 = np.asarray(y).astype(int)
        rng = np.random.default_rng(self.random_state)
        if len(y01) > self.max_samples:
            tr, rest = _stratified_subsample(y01, self.max_samples, rng)
        else:
            tr, rest = np.arange(len(y01)), np.array([], dtype=int)

        Xs, ys = X[tr], np.where(y01[tr] == 1, 1.0, -1.0)
        alpha = self._smo(self.kernel.compute(Xs, Xs), ys)

        sv = alpha > 1e-8
        self.support_vectors_ = Xs[sv]
        self.dual_coef_ = alpha[sv] * ys[sv]  # alpha_i * y_i
        self.support_ = tr[sv]

        if len(rest):  # kalibrasi pada baris yang tidak dipakai melatih SVM
            cal = rest if len(rest) <= self.n_calib else rng.choice(rest, self.n_calib, replace=False)
        else:  # shortcut: data kecil, kalibrasi pada data latih (bias optimis); pakai CV bila perlu
            cal = tr
        self.platt_ = _fit_platt(self.decision_function(X[cal]), y01[cal])
        return self

    def _smo(self, K, y):
        """Minimalkan 1/2 a^T Q a - e^T a, 0 <= a <= C, y^T a = 0, dengan Q_ij = y_i y_j K_ij."""
        n, C = len(y), self.C
        alpha = np.zeros(n)
        G = -np.ones(n)  # gradien Q a - e
        diagK = np.diag(K).copy()
        self.converged_ = False
        for it in range(self.max_iter):
            up = ((y > 0) & (alpha < C)) | ((y < 0) & (alpha > 0))
            low = ((y > 0) & (alpha > 0)) | ((y < 0) & (alpha < C))
            score = -y * G
            i = np.flatnonzero(up)[np.argmax(score[up])]
            j = np.flatnonzero(low)[np.argmin(score[low])]
            gap = score[i] - score[j]
            if gap < self.tol:  # KKT terpenuhi dalam toleransi
                self.converged_ = True
                break
            # Langkah t sepanjang a_i += y_i t, a_j -= y_j t (menjaga y^T a), lalu dipotong ke kotak [0, C].
            eta = max(diagK[i] + diagK[j] - 2 * K[i, j], 1e-12)
            t = min(gap / eta,
                    C - alpha[i] if y[i] > 0 else alpha[i],
                    alpha[j] if y[j] > 0 else C - alpha[j])
            alpha[i] += y[i] * t
            alpha[j] -= y[j] * t
            G += y * t * (K[:, i] - K[:, j])
        self.n_iter_ = it + 1
        self.kkt_gap_ = gap

        # b dari support vector bebas (0 < a < C): y_t f(x_t) = 1  =>  b = -y_t G_t
        free = (alpha > 1e-8) & (alpha < C - 1e-8)
        self.b_ = (-y * G)[free].mean() if free.any() else (score[up].max() + score[low].min()) / 2
        return alpha

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return self.kernel.compute(X, self.support_vectors_) @ self.dual_coef_ + self.b_

    def predict_proba(self, X):
        A, B = self.platt_
        return 1 / (1 + np.exp(np.clip(A * self.decision_function(X) + B, -500, 500)))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def plot_decision_boundary(self, X, y, path, feature_names=("x1", "x2"), title=None, max_points=1500):
        """[BONUS 4] Bidang pemisah f(x)=0, margin f(x)=±1, dan support vector. Model harus dilatih pada 2 fitur."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X, y = np.asarray(X, dtype=float), np.asarray(y).astype(int)
        if X.shape[1] != 2 or self.support_vectors_.shape[1] != 2:
            raise ValueError("plot_decision_boundary butuh model dan X dengan tepat 2 fitur")
        lo, hi = np.percentile(X, [1, 99], axis=0)  # potong outlier agar plot terbaca
        pad = 0.1 * (hi - lo)
        g1, g2 = np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], 300),
                             np.linspace(lo[1] - pad[1], hi[1] + pad[1], 300))
        f = self.decision_function(np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.contourf(g1, g2, f, levels=[-np.inf, 0, np.inf], colors=["#cfe3f5", "#f7d4cf"], alpha=0.6)
        ax.contour(g1, g2, f, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=[1, 2, 1])
        idx = np.random.default_rng(self.random_state).permutation(len(y))[:max_points]
        for c, col, lab in [(0, "#1f77b4", "tidak batal (0)"), (1, "#d62728", "batal (1)")]:
            m = idx[y[idx] == c]
            ax.scatter(X[m, 0], X[m, 1], s=8, c=col, alpha=0.5, label=lab)
        sv = self.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=30, facecolors="none", edgecolors="k", linewidths=0.6,
                   label=f"support vector ({len(sv)})")
        ax.set(xlim=(g1.min(), g1.max()), ylim=(g2.min(), g2.max()), xlabel=feature_names[0],
               ylabel=feature_names[1], title=title or f"SVM {type(self.kernel).__name__}, C={self.C}")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

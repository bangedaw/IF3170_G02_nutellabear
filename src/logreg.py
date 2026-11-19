"""Logistic Regression from scratch (NumPy) dengan regularisasi L1, L2, dan Elastic Net.

Fungsi tujuan (rata-rata log-loss ditambah penalti):

    J(w, b) = (1/n) sum_i [ log(1 + exp(z_i)) - y_i z_i ] + R(w),   z_i = x_i . w + b

    L1          : R(w) = alpha * ||w||_1
    L2          : R(w) = alpha/2 * ||w||_2^2
    Elastic Net : R(w) = alpha * ( l1_ratio * ||w||_1 + (1 - l1_ratio)/2 * ||w||_2^2 )

Bias `b` tidak diregularisasi. Optimasi: gradient descent (batch penuh atau mini-batch). Bagian penalti L1
tidak mulus di 0, jadi dipakai langkah proximal (soft-threshold) setelah langkah gradien. Dengan cara ini
bobot yang tidak berguna bernilai tepat 0, bukan hanya mendekati 0.

Hubungan dengan scikit-learn: objektif sklearn memakai C * sum(loss) + penalti, sehingga alpha = 1 / (C * n).

Referensi:
- M. L. Khodra, "Logistic Regression: What & Why" dan "Stochastic Gradient Ascent", YouTube, 2021 (rujukan kuliah).
- A. Beck & M. Teboulle (2009). A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems.
  SIAM Journal on Imaging Sciences 2(1). (langkah proximal / soft-threshold untuk L1)
- H. Zou & T. Hastie (2005). Regularization and Variable Selection via the Elastic Net.
  Journal of the Royal Statistical Society B 67(2). (bentuk penalti Elastic Net)
"""
import pickle
import shutil

import numpy as np


def sigmoid(z):
    """Sigmoid stabil secara numerik: tidak overflow untuk |z| besar."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def log_loss(y, p, eps=1e-15):
    """Rata-rata binary cross-entropy. `p` dipotong ke [eps, 1-eps] agar log(0) aman."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class Regularizer:
    """Penalti R(w). `grad_smooth` adalah gradien bagian mulus (L2), `prox` langkah proximal bagian L1."""

    def penalty(self, w):
        raise NotImplementedError

    def grad_smooth(self, w):
        raise NotImplementedError

    def prox(self, w, step):
        raise NotImplementedError


def _soft_threshold(w, t):
    return np.sign(w) * np.maximum(np.abs(w) - t, 0.0)


class NoPenalty(Regularizer):
    def penalty(self, w):
        return 0.0

    def grad_smooth(self, w):
        return np.zeros_like(w)

    def prox(self, w, step):
        return w


class L1(Regularizer):
    def __init__(self, alpha):
        self.alpha = alpha

    def penalty(self, w):
        return self.alpha * float(np.abs(w).sum())

    def grad_smooth(self, w):
        return np.zeros_like(w)

    def prox(self, w, step):
        return _soft_threshold(w, step * self.alpha)


class L2(Regularizer):
    def __init__(self, alpha):
        self.alpha = alpha

    def penalty(self, w):
        return 0.5 * self.alpha * float(w @ w)

    def grad_smooth(self, w):
        return self.alpha * w

    def prox(self, w, step):
        return w


class ElasticNet(Regularizer):
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha, self.l1_ratio = alpha, l1_ratio

    def penalty(self, w):
        return self.alpha * (self.l1_ratio * float(np.abs(w).sum())
                             + 0.5 * (1 - self.l1_ratio) * float(w @ w))

    def grad_smooth(self, w):
        return self.alpha * (1 - self.l1_ratio) * w

    def prox(self, w, step):
        return _soft_threshold(w, step * self.alpha * self.l1_ratio)


def make_regularizer(penalty, alpha, l1_ratio=0.5):
    if penalty == "none":
        return NoPenalty()
    if penalty == "l1":
        return L1(alpha)
    if penalty == "l2":
        return L2(alpha)
    if penalty == "elasticnet":
        return ElasticNet(alpha, l1_ratio)
    raise ValueError("penalty harus 'none', 'l1', 'l2', atau 'elasticnet'.")


class LogisticRegression:
    """Klasifikasi biner. Fitur harus numerik, tanpa NaN (pakai tampilan numerik dari preprocessing).

    lr          : laju belajar (fitur terstandarisasi: 0.1 sampai 0.5 biasanya aman).
    n_iter      : jumlah epoch maksimum.
    penalty     : "none", "l1", "l2", atau "elasticnet".
    alpha       : kekuatan regularisasi. l1_ratio: porsi L1 pada elasticnet (0 = L2 murni, 1 = L1 murni).
    tol         : berhenti jika perubahan objektif antar epoch < tol.
    batch_size  : None = batch penuh, bilangan bulat = mini-batch (diacak per epoch dengan random_state).
    record_every: riwayat parameter dicatat tiap sekian epoch (epoch 0 dan terakhir selalu dicatat).

    Setelah fit: coef_ (d,), intercept_, n_iter_, converged_, history_ = {"epoch", "loss", "params"}
    dengan params[:, 0] = bias dan params[:, 1:] = bobot; loss = objektif (log-loss + penalti).
    """

    def __init__(self, lr=0.1, n_iter=2000, penalty="l2", alpha=0.01, l1_ratio=0.5,
                 tol=1e-8, batch_size=None, random_state=42, record_every=1):
        self.lr, self.n_iter, self.penalty, self.alpha = lr, n_iter, penalty, alpha
        self.l1_ratio, self.tol, self.batch_size = l1_ratio, tol, batch_size
        self.random_state, self.record_every = random_state, record_every

    def _validate_params(self):
        make_regularizer(self.penalty, self.alpha, self.l1_ratio)
        if not self.lr > 0:
            raise ValueError("lr harus > 0.")
        if int(self.n_iter) < 1:
            raise ValueError("n_iter harus >= 1.")
        if self.alpha < 0:
            raise ValueError("alpha harus >= 0.")
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio harus di [0, 1].")
        if self.batch_size is not None and int(self.batch_size) < 1:
            raise ValueError("batch_size harus None atau >= 1.")
        if int(self.record_every) < 1:
            raise ValueError("record_every harus >= 1.")

    @staticmethod
    def _check_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X harus 2 dimensi (n_baris, n_fitur).")
        if not np.isfinite(X).all():
            raise ValueError("X mengandung NaN atau inf; imputasi dulu di preprocessing.")
        return X

    def fit(self, X, y):
        self._validate_params()
        X = self._check_X(X)
        y = np.asarray(y).astype(float)
        if y.shape != (len(X),):
            raise ValueError("y harus 1 dimensi dengan panjang sama dengan X.")
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("y harus berisi 0 atau 1.")

        n, d = X.shape
        reg = make_regularizer(self.penalty, self.alpha, self.l1_ratio)
        rng = np.random.default_rng(self.random_state)
        w, b = np.zeros(d), 0.0

        def objective():
            return log_loss(y, sigmoid(X @ w + b)) + reg.penalty(w)

        prev = objective()
        epochs, losses, params = [0], [prev], [np.r_[b, w]]
        self.converged_ = False
        for epoch in range(1, int(self.n_iter) + 1):
            if self.batch_size is None:
                batches = [np.arange(n)]
            else:
                order = rng.permutation(n)
                batches = np.array_split(order, max(1, -(-n // int(self.batch_size))))
            for idx in batches:
                err = sigmoid(X[idx] @ w + b) - y[idx]
                grad_w = X[idx].T @ err / len(idx) + reg.grad_smooth(w)
                w = reg.prox(w - self.lr * grad_w, self.lr)
                b -= self.lr * err.mean()
            obj = objective()
            if not np.isfinite(obj):
                raise ValueError("Loss tidak finite: lr terlalu besar atau fitur belum diskalakan.")
            if epoch % int(self.record_every) == 0:
                epochs.append(epoch)
                losses.append(obj)
                params.append(np.r_[b, w])
            if abs(prev - obj) < self.tol:
                self.converged_ = True
                break
            prev = obj
        if epochs[-1] != epoch:
            epochs.append(epoch)
            losses.append(obj)
            params.append(np.r_[b, w])

        self.coef_, self.intercept_, self.n_iter_ = w, float(b), epoch
        self.n_features_ = d
        self.history_ = {"epoch": np.array(epochs), "loss": np.array(losses), "params": np.vstack(params)}
        return self

    def _check_fitted(self):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Model belum di-fit.")

    def decision_function(self, X):
        self._check_fitted()
        X = self._check_X(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X punya {X.shape[1]} fitur, model dilatih dengan {self.n_features_}.")
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        """Peluang kelas 1, array (m,)."""
        return sigmoid(self.decision_function(X))

    def predict(self, X, threshold=0.5):
        """Kelas 1 jika peluang >= threshold."""
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold harus angka finite dalam [0, 1].")
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"Berkas {path} berisi {type(model).__name__}, bukan {cls.__name__}.")
        return model

    def plot_loss_contour_video(self, X, y, path, fps=20, max_frames=150, max_points=2000,
                                feature_name="x", grid=80):
        """[BONUS 3] Video kontur log-loss pada bidang (theta0, theta1) beserta lintasan parameter saat training.

        Model harus dilatih dengan tepat 1 fitur (theta0 = bias, theta1 = bobot). X, y: data yang sama dengan
        saat fit (disubsampel sampai max_points baris hanya untuk menggambar kontur). Format mp4 bila ffmpeg
        tersedia; jika tidak, GIF. Mengembalikan path berkas yang benar-benar ditulis.
        """
        self._check_fitted()
        if self.n_features_ != 1:
            raise ValueError("Video butuh model dengan tepat 1 fitur (2 parameter: theta0 dan theta1).")
        from matplotlib import animation
        from matplotlib.figure import Figure

        X, y = self._check_X(X), np.asarray(y).astype(float)
        if len(X) > max_points:
            keep = np.random.default_rng(self.random_state).choice(len(X), max_points, replace=False)
            X, y = X[keep], y[keep]
        x = X[:, 0]

        traj = self.history_["params"]
        loss_hist = self.history_["loss"]
        frames = np.unique(np.linspace(0, len(traj) - 1, min(max_frames, len(traj))).round().astype(int))

        lo, hi = traj.min(axis=0), traj.max(axis=0)
        pad = np.maximum(0.3 * (hi - lo), 0.5)
        t0 = np.linspace(lo[0] - pad[0], hi[0] + pad[0], grid)
        t1 = np.linspace(lo[1] - pad[1], hi[1] + pad[1], grid)
        T0, T1 = np.meshgrid(t0, t1)
        Z = np.empty_like(T0)
        for r in range(grid):
            z = T0[r][:, None] + T1[r][:, None] * x[None, :]
            Z[r] = (np.logaddexp(0, z) - y[None, :] * z).mean(axis=1)

        fig = Figure(figsize=(7, 5.5))
        ax = fig.subplots()
        cs = ax.contour(T0, T1, Z, levels=18, cmap="viridis")
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
        ax.set_xlabel(r"$\theta_0$ (bias)")
        ax.set_ylabel(rf"$\theta_1$ (bobot {feature_name})")
        (line,) = ax.plot([], [], "-", color="#c0392b", lw=1.8)
        (dot,) = ax.plot([], [], "o", color="#c0392b", ms=6)
        ax.plot(traj[0, 0], traj[0, 1], "ks", ms=6, label="awal")
        ax.legend(loc="upper right")
        title = ax.set_title("")

        def draw(k):
            i = frames[k]
            line.set_data(traj[: i + 1, 0], traj[: i + 1, 1])
            dot.set_data([traj[i, 0]], [traj[i, 1]])
            title.set_text(f"epoch {int(self.history_['epoch'][i])}, objektif {loss_hist[i]:.4f}")
            return line, dot, title

        anim = animation.FuncAnimation(fig, draw, frames=len(frames), blit=False)
        path = str(path)
        if path.lower().endswith(".mp4") and shutil.which("ffmpeg"):
            writer = animation.FFMpegWriter(fps=fps)
        else:
            if path.lower().endswith(".mp4"):
                path = path[:-4] + ".gif"
            writer = animation.PillowWriter(fps=fps)
        anim.save(path, writer=writer, dpi=110)
        return path

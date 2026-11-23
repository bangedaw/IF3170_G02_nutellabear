"""Optimizer tambahan untuk LogisticRegression (bonus 6): FISTA dan Adam. Kelas induk di logreg.py tidak diubah.

- FISTA: gradient descent proximal yang dipercepat (momentum Nesterov). Laju konvergensi objektif O(1/k^2), dibanding
  O(1/k) untuk gradient descent proximal biasa (ISTA) yang dipakai kelas induk. Berlaku untuk semua penalti, termasuk
  L1 dan Elastic Net. Ukuran langkah 1/L dengan L konstanta Lipschitz gradien log-loss: lambda_max(X^T X / n) / 4
  ditambah bagian L2, dihitung dengan iterasi pangkat.
  A. Beck, M. Teboulle (2009). A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems.
  SIAM Journal on Imaging Sciences 2(1):183-202.
- Adam: laju belajar adaptif per parameter dari estimasi momen pertama dan kedua gradien, untuk mini-batch.
  D. P. Kingma, J. Ba (2015). Adam: A Method for Stochastic Optimization. ICLR 2015, https://arxiv.org/abs/1412.6980.
  Hanya untuk penalti mulus ("none", "l2"); bentuk proximal Adam untuk L1 tidak ada di makalah itu.
"""
import numpy as np

from logreg import LogisticRegression, log_loss, make_regularizer, sigmoid


def lipschitz_constant(X, l2=0.0, n_iter=100, seed=0):
    """Batas atas konstanta Lipschitz gradien rata-rata log-loss (bias disertakan sebagai kolom satu)."""
    Z = np.column_stack([np.ones(len(X)), X])
    v = np.random.default_rng(seed).normal(size=Z.shape[1])
    for _ in range(n_iter):
        v = Z.T @ (Z @ v)
        v /= np.linalg.norm(v)
    return float(v @ (Z.T @ (Z @ v))) / (4 * len(X)) + l2


class OptimizedLogisticRegression(LogisticRegression):
    """LogisticRegression dengan optimizer "gd" (perilaku induk), "fista", atau "adam".
    lr=None pada FISTA memakai 1/L. Adam: beta1, beta2, eps sesuai makalah; batch_size None berarti batch penuh."""

    def __init__(self, optimizer="fista", lr=None, n_iter=2000, penalty="l2", alpha=0.01, l1_ratio=0.5, tol=1e-8,
                 batch_size=None, random_state=42, record_every=1, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr=0.1 if lr is None else lr, n_iter=n_iter, penalty=penalty, alpha=alpha,
                         l1_ratio=l1_ratio, tol=tol, batch_size=batch_size, random_state=random_state,
                         record_every=record_every)
        self.optimizer, self.auto_lr = optimizer, lr is None
        self.beta1, self.beta2, self.eps = beta1, beta2, eps

    def fit(self, X, y):
        if self.optimizer == "gd":
            return super().fit(X, y)
        if self.optimizer == "adam" and self.penalty not in ("none", "l2"):
            raise ValueError("Adam di sini hanya untuk penalti 'none' atau 'l2'.")
        if self.optimizer not in ("fista", "adam"):
            raise ValueError("optimizer harus 'gd', 'fista', atau 'adam'.")
        self._validate_params()
        X = self._check_X(X)
        y = np.asarray(y).astype(float)
        if y.shape != (len(X),) or not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("y harus 1 dimensi berisi 0/1 dengan panjang sama dengan X.")
        n, d = X.shape
        reg = make_regularizer(self.penalty, self.alpha, self.l1_ratio)
        l2 = {"l2": self.alpha, "elasticnet": self.alpha * (1 - self.l1_ratio)}.get(self.penalty, 0.0)
        if self.optimizer == "fista" and self.auto_lr:
            self.lr = 1.0 / lipschitz_constant(X, l2)
        theta = np.zeros(d + 1)

        def objective(t):
            return log_loss(y, sigmoid(X @ t[1:] + t[0])) + reg.penalty(t[1:])

        def gradient(t, idx):
            err = sigmoid(X[idx] @ t[1:] + t[0]) - y[idx]
            return np.r_[err.mean(), X[idx].T @ err / len(idx) + reg.grad_smooth(t[1:])]

        prev = objective(theta)
        epochs, losses, params = [0], [prev], [theta.copy()]
        self.converged_ = False
        rng = np.random.default_rng(self.random_state)
        momentum_t, previous = 1.0, theta.copy()
        m, v, step = np.zeros(d + 1), np.zeros(d + 1), 0
        for epoch in range(1, int(self.n_iter) + 1):
            if self.optimizer == "fista":
                t_next = (1 + np.sqrt(1 + 4 * momentum_t ** 2)) / 2
                z = theta + (momentum_t - 1) / t_next * (theta - previous)
                g = gradient(z, np.arange(n))
                new = z - self.lr * g
                new[1:] = reg.prox(new[1:], self.lr)
                previous, theta, momentum_t = theta, new, t_next
            else:
                if self.batch_size is None:
                    batches = [np.arange(n)]
                else:
                    batches = np.array_split(rng.permutation(n), max(1, -(-n // int(self.batch_size))))
                for idx in batches:
                    step += 1
                    g = gradient(theta, idx)
                    m = self.beta1 * m + (1 - self.beta1) * g
                    v = self.beta2 * v + (1 - self.beta2) * g * g
                    theta = theta - self.lr * (m / (1 - self.beta1 ** step)) / (np.sqrt(v / (1 - self.beta2 ** step)) + self.eps)
            obj = objective(theta)
            if not np.isfinite(obj):
                raise ValueError("Loss tidak finite: lr terlalu besar atau fitur belum diskalakan.")
            if epoch % int(self.record_every) == 0:
                epochs.append(epoch)
                losses.append(obj)
                params.append(theta.copy())
            if abs(prev - obj) < self.tol:
                self.converged_ = True
                break
            prev = obj
        if epochs[-1] != epoch:
            epochs.append(epoch)
            losses.append(obj)
            params.append(theta.copy())
        self.coef_, self.intercept_, self.n_iter_, self.n_features_ = theta[1:], float(theta[0]), epoch, d
        self.history_ = {"epoch": np.array(epochs), "loss": np.array(losses), "params": np.vstack(params)}
        return self

"""Tes SVM: jalankan `python -m pytest tests/test_svm.py` atau `python tests/test_svm.py` dari root repo."""
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from svm import SVM, GaussianKernel, LinearKernel, PolynomialKernel  # noqa: E402

X, y = make_classification(n_samples=600, n_features=6, random_state=42)


def test_kernels_match_sklearn():
    A, B = X[:50], X[50:90]
    assert np.allclose(LinearKernel().compute(A, B), linear_kernel(A, B))
    assert np.allclose(PolynomialKernel(3, 0.2, 1.0).compute(A, B), polynomial_kernel(A, B, 3, 0.2, 1.0))
    assert np.allclose(GaussianKernel(0.3).compute(A, B), rbf_kernel(A, B, 0.3))


def test_matches_sklearn_svc():
    for kernel, sk in [(LinearKernel(), dict(kernel="linear")),
                       (PolynomialKernel(3, 0.2, 1.0), dict(kernel="poly", degree=3, gamma=0.2, coef0=1.0)),
                       (GaussianKernel(0.3), dict(kernel="rbf", gamma=0.3))]:
        m = SVM(kernel, C=1.0).fit(X, y)
        ref = SVC(C=1.0, tol=1e-3, **sk).fit(X, y)
        assert m.converged_
        assert np.mean(np.sign(m.decision_function(X)) == np.sign(ref.decision_function(X))) > 0.98
        assert abs(len(m.support_) - len(ref.support_)) <= 0.05 * len(ref.support_)


def test_interface_and_save_load():
    m = SVM(GaussianKernel(0.3), max_samples=300).fit(X, y)  # jalur subsampel + kalibrasi pada sisa
    p = m.predict_proba(X)
    assert p.shape == (len(X),) and (p >= 0).all() and (p <= 1).all()
    assert set(np.unique(m.predict(X))) <= {0, 1}
    with tempfile.TemporaryDirectory() as d:
        m.save(Path(d) / "svm.pkl")
        assert np.array_equal(SVM.load(Path(d) / "svm.pkl").predict_proba(X), p)


def test_plot_decision_boundary():
    X2 = X[:, :2]
    m = SVM(LinearKernel(), max_samples=300).fit(X2, y)
    with tempfile.TemporaryDirectory() as d:
        m.plot_decision_boundary(X2, y, Path(d) / "b.png")
        assert (Path(d) / "b.png").stat().st_size > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

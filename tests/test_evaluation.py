"""Tes evaluasi: `python -m pytest tests/test_evaluation.py` dari root repo."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import (accuracy, confusion_matrix, cross_validate, f1, precision, recall,  # noqa: E402
                        roc_auc, stratified_kfold)

rng = np.random.default_rng(0)
y = rng.integers(0, 2, 500)
score = np.round(rng.random(500) + 0.3 * y, 1)  # dibulatkan agar banyak skor kembar
y_pred = (score >= 0.6).astype(int)


def test_metrics_match_sklearn():
    assert (confusion_matrix(y, y_pred) == metrics.confusion_matrix(y, y_pred)).all()
    assert np.isclose(accuracy(y, y_pred), metrics.accuracy_score(y, y_pred))
    assert np.isclose(precision(y, y_pred), metrics.precision_score(y, y_pred))
    assert np.isclose(recall(y, y_pred), metrics.recall_score(y, y_pred))
    assert np.isclose(f1(y, y_pred), metrics.f1_score(y, y_pred))
    assert np.isclose(roc_auc(y, score), metrics.roc_auc_score(y, score))


def test_stratified_kfold():
    yy = np.r_[np.zeros(670), np.ones(330)].astype(int)
    folds = stratified_kfold(yy, k=5, seed=42)
    all_val = np.sort(np.concatenate([va for _, va in folds]))
    assert (all_val == np.arange(len(yy))).all()  # tiap baris tepat sekali jadi validasi
    for tr, va in folds:
        assert not np.intersect1d(tr, va).size
        assert abs(yy[va].mean() - 0.33) < 0.01


def test_cross_validate_runs():
    class Constant:  # model palsu: selalu prevalensi kelas
        def fit(self, X, y):
            self.p = np.mean(y)
            return self

        def predict_proba(self, X):
            return np.full(len(X), self.p)

    df = pd.DataFrame({"x": rng.random(200), "is_canceled": rng.integers(0, 2, 200)})
    prep = lambda tr, va: (tr[["x"]].values, tr.is_canceled.values, va[["x"]].values, va.is_canceled.values)  # noqa: E731
    out = cross_validate(Constant, prep, df, k=4)
    assert list(out.index) == [0, 1, 2, 3, "mean"]
    assert np.isclose(out.loc["mean", "roc_auc"], 0.5)

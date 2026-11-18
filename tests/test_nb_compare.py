import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from naive_bayes import NaiveBayes
from nb_compare import SklearnCategoricalNB, exact_equivalence

rng = np.random.default_rng(42)
N = 800
y_all = rng.integers(0, 2, N)
df_all = pd.DataFrame({
    "num": rng.normal(y_all * 1.0, 1.0),
    "cat": np.where(rng.random(N) < 0.8, np.where(y_all == 1, "a", "b"), "c"),
    "few": rng.integers(0, 3, N),
})
X_tr, y_tr, X_va = df_all[:600], y_all[:600], df_all[600:]


def test_nb_core_matches_categorical_nb_given_same_bins():
    r = exact_equivalence(X_tr, y_tr, X_va, NaiveBayes)
    assert r["max_abs_diff"] < 1e-9
    assert r["same_prediction"] == 1.0


def test_sklearn_wrapper_handles_missing_and_unseen():
    Xn = X_tr.copy()
    Xn.loc[Xn.index[:50], "num"] = np.nan
    Xn.loc[Xn.index[50:90], "cat"] = np.nan
    new = pd.DataFrame({"num": [0.0, np.nan], "cat": ["zzz", np.nan], "few": [1, 99]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = SklearnCategoricalNB().fit(Xn, y_tr).predict_proba(new)
    assert p.shape == (2,) and np.all((p >= 0) & (p <= 1))

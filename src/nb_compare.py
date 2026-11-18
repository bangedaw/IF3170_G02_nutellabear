import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder


class SklearnCategoricalNB:
    def __init__(self, alpha=1.0, n_bins=20, max_categorical=10, class_prior=None):
        self.alpha, self.n_bins, self.max_categorical, self.class_prior = alpha, n_bins, max_categorical, class_prior

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.encoders_ = {}
        for c in self.columns_:
            col = X[c]
            if pd.api.types.is_numeric_dtype(col) and col.nunique() > self.max_categorical:
                enc = KBinsDiscretizer(self.n_bins, encode="ordinal", strategy="quantile")
                enc.fit(col.dropna().to_numpy(dtype=float).reshape(-1, 1))
            else:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                enc.fit(col.dropna().astype(object).to_numpy().reshape(-1, 1))
            self.encoders_[c] = enc
        self.model_ = CategoricalNB(alpha=self.alpha, class_prior=self.class_prior)
        self.model_.fit(self._codes(X), np.asarray(y).astype(int))
        return self

    def _codes(self, X):
        out = np.zeros((len(X), len(self.columns_)), dtype=int)
        for j, c in enumerate(self.columns_):
            col, enc = X[c], self.encoders_[c]
            ok = col.notna().to_numpy()
            if ok.any():
                v = col[ok].to_numpy(dtype=float if isinstance(enc, KBinsDiscretizer) else object).reshape(-1, 1)
                code = enc.transform(v).ravel().astype(int) + 1
                out[ok, j] = np.where(code < 0, 0, code)
        return out

    def predict_proba(self, X):
        return self.model_.predict_proba(self._codes(pd.DataFrame(X)))[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def exact_equivalence(X_tr, y_tr, X_va, nb, n_bins=20):
    full_tr, full_va = ~X_tr.isna().any(axis=1).to_numpy(), ~X_va.isna().any(axis=1).to_numpy()
    X_tr, y_tr, X_va = X_tr[full_tr], np.asarray(y_tr)[full_tr], X_va[full_va]
    scratch = nb(n_bins=n_bins).fit(X_tr, y_tr)

    def codes(X):
        return np.column_stack([scratch._codes(c, X[c]) for c in scratch.columns_])

    sk = CategoricalNB(alpha=scratch.alpha, class_prior=np.exp(scratch.log_prior_),
                       min_categories=[scratch.log_lik_[c].shape[1] for c in scratch.columns_])
    sk.fit(codes(X_tr), np.asarray(y_tr).astype(int))
    p_s, p_k = scratch.predict_proba(X_va), sk.predict_proba(codes(X_va))[:, 1]
    return {"n_train": len(X_tr), "n_val": len(X_va), "max_abs_diff": float(np.abs(p_s - p_k).max()),
            "same_prediction": float(((p_s >= .5) == (p_k >= .5)).mean())}

"""Tuning KNN: pembobotan kolom lewat mutual information, fitur siklik/log, dan pencarian acak dengan CV.

Semua yang belajar dari data (statistik scaler, batas clip, MI) di-fit hanya pada train fold. KNN tetap satu model
(bukan ensemble). Referensi KNN: P. Cunningham & S. J. Delany, arXiv:2004.04523, 2020.
"""
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from evaluation import scores, stratified_kfold
from knn import KNN
from preprocessing import (TARGET, CategoricalEncoder, Clipper, Imputer, PreparationPipeline, Scaler,
                           construct_features, drop_duplicates_and_invalid)

KS = (5, 10, 15, 21, 31, 41, 61, 81, 101)
WEIGHTINGS = ("uniform", "distance")
PIPELINE_KEYS = ("scaler", "cyclic", "log")
MODEL_KEYS = ("metric", "p", "gamma", "top_k", "onehot_factor")
BASELINE = {"scaler": "standard", "cyclic": False, "log": False, "metric": "manhattan", "p": 1,
            "gamma": 0.0, "top_k": 0, "onehot_factor": 1.0}
SPACE = {"scaler": ["standard", "minmax"], "cyclic": [False, True], "log": [False, True],
         "metric": [("manhattan", 1), ("euclidean", 2)], "gamma": [0.0, 0.5, 1.0, 2.0],
         "top_k": [0, 8, 12, 16, 20, 24], "onehot_factor": [1.0, 0.5, 0.25]}


def cyclic_month(df):
    df = df.copy()
    angle = 2 * np.pi * (df["arrival_month"] - 1) / 12
    df["month_sin"], df["month_cos"] = np.sin(angle), np.cos(angle)
    return df.drop(columns=["arrival_month"])


def log_skewed(df, cols=("days_before_arrival", "nightly_rate_idr", "nightly_rate_per_guest")):
    df = df.copy()
    for c in cols:
        df[c] = np.log1p(df[c].clip(lower=0))
    return df


def build_pipeline(scaler="standard", cyclic=False, log=False):
    steps = [construct_features]
    if log:
        steps.append(log_skewed)
    if cyclic:
        steps.append(cyclic_month)
    steps += [Imputer(), Clipper(), CategoricalEncoder(), Scaler(scaler)]
    return PreparationPipeline(steps)


def column_mutual_information(X, y, n_bins=20):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    p_y = np.bincount(y, minlength=2) / len(y)
    mi = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        values = np.unique(col)
        if len(values) > n_bins:
            edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1]))
            code = np.searchsorted(edges, col, side="right")
        else:
            code = np.searchsorted(values, col)
        joint = np.bincount(code * 2 + y, minlength=2 * (code.max() + 1)).reshape(-1, 2) / len(y)
        expected = joint.sum(1, keepdims=True) * p_y
        nonzero = joint > 0
        mi[j] = float((joint[nonzero] * np.log2(joint[nonzero] / expected[nonzero])).sum())
    return mi


def column_scale(mi, names, gamma=0.0, top_k=0, onehot_factor=1.0):
    mi = np.asarray(mi, dtype=float)
    weight = np.ones(len(mi)) if gamma == 0 or mi.max() <= 0 else (mi / mi.max()) ** gamma
    if top_k:
        keep = np.zeros(len(mi))
        keep[np.argsort(-mi, kind="stable")[:top_k]] = 1
        weight = weight * keep
    onehot = np.array(["=" in str(n) for n in names])
    return np.where(onehot, weight * onehot_factor, weight)


class WeightedKNN:
    """KNN dengan kolom dikali bobot. Pada jarak Minkowski orde p, kolom dikali w setara bobot w**p pada selisih
    kolom itu. Antarmuka sama dengan KNN: fit, predict_proba, predict(threshold), save, load."""

    def __init__(self, k=15, metric="manhattan", p=1, weights="distance", gamma=0.0, top_k=0,
                 onehot_factor=1.0, n_bins=20, batch_size=256):
        self.k, self.metric, self.p, self.weights = k, metric, p, weights
        self.gamma, self.top_k, self.onehot_factor = gamma, top_k, onehot_factor
        self.n_bins, self.batch_size = n_bins, batch_size

    def _params(self):
        return {"k": self.k, "metric": self.metric, "p": self.p, "weights": self.weights, "gamma": self.gamma,
                "top_k": self.top_k, "onehot_factor": self.onehot_factor, "n_bins": self.n_bins,
                "batch_size": self.batch_size}

    def _scaled(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_] * self.scale_

    def fit(self, X, y):
        names = list(pd.DataFrame(X).columns)
        y = np.asarray(y).astype(int)
        self.mi_ = column_mutual_information(X, y, self.n_bins)
        scale = column_scale(self.mi_, names, self.gamma, self.top_k, self.onehot_factor)
        self.keep_ = np.flatnonzero(scale > 0)
        self.scale_ = scale[self.keep_]
        self.knn_ = KNN(self.k, self.metric, self.p, self.weights, self.batch_size).fit(self._scaled(X), y)
        return self

    def kneighbors(self, X):
        return self.knn_.kneighbors(self._scaled(X))

    def predict_proba(self, X):
        return self.knn_.predict_proba(self._scaled(X))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        with open(path, "wb") as f:
            np.savez_compressed(f, X=self.knn_.X_, y=self.knn_.y_, keep=self.keep_, scale=self.scale_,
                                mi=self.mi_, params=np.array(json.dumps(self._params())))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(**json.loads(str(data["params"])))
            model.mi_, model.keep_, model.scale_ = data["mi"], data["keep"], data["scale"]
            model.knn_ = KNN(model.k, model.metric, model.p, model.weights, model.batch_size)
            model.knn_.X_, model.knn_.y_ = data["X"], data["y"]
        return model


def proba_from_neighbors(dist, labels, k, weights="distance"):
    d, lab = dist[:, :k], labels[:, :k]
    if weights == "uniform":
        return lab.mean(1)
    with np.errstate(divide="ignore"):
        w = 1 / d
    exact = np.isinf(w)
    w[exact.any(1)] = exact[exact.any(1)]
    return (w * lab).sum(1) / w.sum(1)


def random_configs(n, seed=42):
    rng = np.random.default_rng(seed)
    configs = [dict(BASELINE)]
    seen = {tuple(sorted(BASELINE.items()))}
    while len(configs) < n + 1:
        metric, p = SPACE["metric"][rng.integers(len(SPACE["metric"]))]
        config = {key: SPACE[key][rng.integers(len(SPACE[key]))] for key in SPACE if key != "metric"}
        config.update(metric=metric, p=p)
        key = tuple(sorted(config.items()))
        if key not in seen:
            seen.add(key)
            configs.append(config)
    return configs


def evaluate_fold(df, config, fold, ks=KS, k_folds=5, seed=42):
    train_idx, val_idx = stratified_kfold(df[TARGET].to_numpy(), k_folds, seed)[fold]
    train, _ = drop_duplicates_and_invalid(df.iloc[train_idx])
    pipeline = build_pipeline(**{k: config[k] for k in PIPELINE_KEYS}).fit(train)
    X_tr, y_tr = pipeline.transform(train)
    X_va, y_va = pipeline.transform(df.iloc[val_idx])
    model = WeightedKNN(k=max(ks), **{k: config[k] for k in MODEL_KEYS}).fit(X_tr, y_tr)
    dist, idx = model.kneighbors(X_va)
    labels = model.knn_.y_[idx]
    rows = []
    for k in ks:
        for weights in WEIGHTINGS:
            result = scores(y_va, proba_from_neighbors(dist, labels, k, weights))
            rows.append({**config, "k": k, "weights": weights, "fold": fold, **result})
    return rows


def run_search(df, configs, n_jobs=8, ks=KS, k_folds=5, seed=42):
    tasks = [(config, fold) for config in configs for fold in range(k_folds)]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(evaluate_fold, df, config, fold, ks, k_folds, seed) for config, fold in tasks]
        rows = [row for future in futures for row in future.result()]
    return pd.DataFrame(rows)


def summarize(results):
    keys = [c for c in results.columns if c in (*PIPELINE_KEYS, *MODEL_KEYS, "k", "weights")]
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    grouped = results.groupby(keys)
    out = grouped[metrics].mean()
    out["accuracy_std"] = grouped["accuracy"].std()
    return out.reset_index().sort_values(["accuracy", "f1"], ascending=False).reset_index(drop=True)

"""Tuning DTL: cost-complexity pruning, seleksi fitur lewat mutual information, dan pencarian acak dengan CV.

Memakai DecisionTree apa adanya. Pemangkasan dilakukan di luar kelas pada pohon yang sudah dilatih. Semua yang belajar
dari data (MI untuk seleksi fitur, pohon) di-fit hanya pada train fold. Satu pohon, bukan ensemble.
Cost-complexity pruning: L. Breiman, J. Friedman, R. Olshen, C. Stone, "Classification and Regression Trees", 1984
(rujukan dari SPEC.md, belum diverifikasi ulang).
Penalti MDL untuk tes numerik: J. R. Quinlan, "Improved Use of Continuous Attributes in C4.5", Journal of Artificial
Intelligence Research 4 (1996) 77-90, https://arxiv.org/abs/cs/9603103.
"""
import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from decision_tree import DecisionTree, TreeNode, _entropy_counts, _Split
from evaluation import scores, stratified_kfold
from naive_bayes import NaiveBayes
from nb_graph import feature_class_mi
from preprocessing import (CATEGORICAL, TARGET, Imputer, PreparationPipeline, construct_features,
                           drop_duplicates_and_invalid)

ALPHAS = (0, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 4e-3)
SPACE = {"derived": [True, False], "max_depth": [20, 30], "min_samples_leaf": [1, 2, 5], "min_gain": [1e-12, 1e-4],
         "month_cat": [False, True], "top_k": [0, 14, 12, 10]}
BASELINE = {"derived": True, "max_depth": 20, "min_samples_leaf": 1, "min_gain": 1e-12, "month_cat": False, "top_k": 0}
BEST = {"derived": False, "max_depth": 20, "min_samples_leaf": 1, "min_gain": 1e-4, "month_cat": False, "top_k": 14,
        "alpha": 5e-5}
MODEL_KEYS = ("max_depth", "min_samples_leaf", "min_gain", "month_cat", "top_k")
BEST_MDL = {"mdl": "gain", "max_depth": 20, "min_samples_leaf": 1, "min_gain": 1e-4, "month_cat": False, "top_k": 14,
            "alpha": 5e-5}
BEST_MDL_THRESHOLD = 0.55


class MDLDecisionTree(DecisionTree):
    """DecisionTree dengan penalti MDL Quinlan (1996) pada tes numerik: gain dikurangi log2(N - 1) / |D|, N jumlah
    nilai berbeda dan |D| bobot baris yang diketahui di simpul. Atribut tanpa sisa gain positif tidak dipakai.
    threshold_by="gain" memilih ambang dengan gain (seperti C4.5 rilis 8), "ratio" dengan gain ratio terpenalti."""

    def __init__(self, *args, threshold_by="gain", **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold_by = threshold_by

    def _best_numeric(self, x, y, weights, feature):
        x = np.asarray(x, dtype=float)
        known = ~np.isnan(x)
        order = np.argsort(x[known], kind="stable")
        values, labels, w = x[known][order], y[known][order], weights[known][order]
        if len(values) < 2:
            return None
        positions = np.flatnonzero(values[:-1] < values[1:])
        if not len(positions):
            return None
        cumulative = np.cumsum(w)
        positive = np.cumsum(w * labels)
        left_w, left_pos = cumulative[positions], positive[positions]
        right_w, right_pos = cumulative[-1] - left_w, positive[-1] - left_pos
        known_w, missing_w = cumulative[-1], weights[~known].sum()
        total = known_w + missing_w
        masses = np.column_stack((left_w, right_w))
        probabilities = masses / known_w
        valid = np.all(probabilities * total >= self.min_samples_leaf - 1e-12, axis=1)
        left = np.column_stack((left_w - left_pos, left_pos))
        right = np.column_stack((right_w - right_pos, right_pos))
        parent_h = _entropy_counts([known_w - positive[-1], positive[-1]])
        gain = np.maximum(0.0, known_w / total *
                          (parent_h - (left_w * _entropy_counts(left)
                                       + right_w * _entropy_counts(right)) / known_w))
        split_info = _entropy_counts(np.column_stack((masses, np.full(len(masses), missing_w))))
        penalised = gain - np.log2(len(positions)) / known_w
        ratios = np.divide(penalised, split_info, out=np.zeros_like(gain), where=split_info > 1e-12)
        valid &= (penalised > self.min_gain) & (split_info > 1e-12)
        if not valid.any():
            return None
        score = gain if self.threshold_by == "gain" else ratios
        best = int(np.argmax(np.where(valid, score, -np.inf)))
        a, b = values[positions[best]], values[positions[best] + 1]
        threshold = a / 2.0 + b / 2.0
        if not a <= threshold < b:
            threshold = a
        return _Split(feature, float(threshold), None, float(penalised[best]), float(ratios[best]),
                      probabilities[best])


def count_nodes(node):
    return 1 + sum(count_nodes(child) for child in node.children.values())


def prune_node(node, alpha, total):
    """Pohon bagian dengan R(T) + alpha * jumlah daun minimum. R memakai galat kelas mayoritas dibagi bobot total."""
    counts = node.class_counts
    leaf_cost = min(counts[0], counts[1]) / total + alpha
    if node.is_leaf():
        return leaf_cost, node
    children, subtree_cost = {}, 0.0
    for key, child in node.children.items():
        cost, new_child = prune_node(child, alpha, total)
        children[key] = new_child
        subtree_cost += cost
    if leaf_cost <= subtree_cost:
        return leaf_cost, TreeNode(counts, node.n_samples, node.depth)
    return subtree_cost, replace(node, children=children)


def prune(tree, alpha):
    """Salinan dangkal DecisionTree dengan root terpangkas. alpha 0 atau None mengembalikan pohon yang sama."""
    if not alpha:
        return tree
    pruned = copy.copy(tree)
    pruned.root = prune_node(tree.root, alpha, tree.root.class_counts.sum())[1]
    return pruned


def tuned_pipeline(derived=False, impute=False):
    """impute=True mengisi nilai hilang dengan median/modus data latih, menggantikan bobot pecahan C4.5."""
    steps = [construct_features] if derived else []
    if impute:
        steps.append(Imputer(indicator_cols=()))
    return PreparationPipeline(steps)


class PrunedDecisionTree:
    """DecisionTree dengan seleksi top_k fitur (MI dari model Naive Bayes, tanpa label uji) lalu cost-complexity
    pruning. Antarmuka: fit, predict_proba, predict(threshold), save, load. X: tampilan pohon (DataFrame)."""

    def __init__(self, max_depth=20, min_samples_leaf=1, min_gain=1e-4, top_k=14, alpha=5e-5, month_cat=False,
                 min_samples_split=2, mdl="off"):
        self.max_depth, self.min_samples_leaf, self.min_gain = max_depth, min_samples_leaf, min_gain
        self.top_k, self.alpha, self.month_cat, self.min_samples_split = top_k, alpha, month_cat, min_samples_split
        self.mdl = mdl

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y).astype(int)
        if self.top_k:
            self.columns_ = list(feature_class_mi(NaiveBayes().fit(X, y)).index[:self.top_k])
        else:
            self.columns_ = list(X.columns)
        categorical = set(CATEGORICAL) | ({"arrival_month"} if self.month_cat else set())
        indices = [i for i, c in enumerate(self.columns_) if c in categorical]
        params = dict(categorical_features=indices, max_depth=self.max_depth,
                      min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                      min_gain=self.min_gain)
        tree = DecisionTree(**params) if self.mdl == "off" else MDLDecisionTree(threshold_by=self.mdl, **params)
        self.full_tree_ = tree.fit(X[self.columns_], y)
        self.model_ = prune(self.full_tree_, self.alpha)
        self.n_nodes_ = count_nodes(self.model_.root)
        return self

    def predict_proba_at(self, X, alpha):
        return prune(self.full_tree_, alpha).predict_proba(pd.DataFrame(X)[self.columns_])

    def predict_proba(self, X):
        return self.model_.predict_proba(pd.DataFrame(X)[self.columns_])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        self.model_.selected_columns_ = self.columns_
        self.model_.save(path)

    @classmethod
    def load(cls, path):
        tree = DecisionTree.load(path)
        model = cls()
        model.model_, model.columns_ = tree, tree.selected_columns_
        model.n_nodes_ = count_nodes(tree.root)
        return model


def random_configs(n, seed=7):
    rng = np.random.default_rng(seed)
    configs, seen = [dict(BASELINE)], {tuple(sorted(BASELINE.items()))}
    while len(configs) < n + 1:
        config = {key: values[rng.integers(len(values))] for key, values in SPACE.items()}
        key = tuple(sorted(config.items()))
        if key not in seen:
            seen.add(key)
            configs.append(config)
    return configs


def evaluate_fold(df, config, fold, alphas=ALPHAS, k_folds=5, seed=42):
    train_idx, val_idx = stratified_kfold(df[TARGET].to_numpy(), k_folds, seed)[fold]
    train, _ = drop_duplicates_and_invalid(df.iloc[train_idx])
    pipeline = tuned_pipeline(config["derived"]).fit(train)
    X_tr, y_tr = pipeline.transform(train)
    X_va, y_va = pipeline.transform(df.iloc[val_idx])
    model = PrunedDecisionTree(alpha=0, **{k: config[k] for k in MODEL_KEYS}).fit(X_tr, y_tr)
    rows = []
    for alpha in alphas:
        proba = model.predict_proba_at(X_va, alpha)
        nodes = count_nodes(prune(model.full_tree_, alpha).root)
        rows.append({**config, "alpha": alpha, "fold": fold, "nodes": nodes, **scores(y_va, proba)})
    return rows


def run_search(df, configs, n_jobs=6, alphas=ALPHAS, k_folds=5, seed=42):
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(evaluate_fold, df, config, fold, alphas, k_folds, seed)
                   for config in configs for fold in range(k_folds)]
        return pd.DataFrame([row for future in futures for row in future.result()])


def summarize(results):
    keys = [c for c in results.columns if c in (*SPACE, "alpha")]
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    grouped = results.groupby(keys)
    out = grouped[metrics].mean()
    out["accuracy_std"] = grouped["accuracy"].std()
    return out.reset_index().sort_values(["accuracy", "f1"], ascending=False).reset_index(drop=True)

from dataclasses import dataclass, field
from numbers import Integral, Real
import pickle

import numpy as np


def _entropy_counts(counts):
    """Entropy sepanjang sumbu terakhir; kelompok kosong bernilai nol."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    logs = np.zeros_like(p)
    np.log2(p, out=logs, where=p > 0)
    return -(p * logs).sum(axis=-1)


def _labels_weights(y, sample_weight=None):
    """Validasi label biner dan bobot sampel nonnegatif."""
    y = np.asarray(y)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("y harus array satu dimensi berisi label 0/1.")
    weights = (np.ones(len(y)) if sample_weight is None
               else np.asarray(sample_weight, dtype=float))
    if (weights.shape != y.shape or not np.isfinite(weights).all()
            or (weights < 0).any()):
        raise ValueError("Bobot harus berukuran sama dengan y, finite, dan nonnegatif.")
    return y.astype(np.int64), weights


def entropy(y, sample_weight=None):
    """Entropy label biner, opsional dengan bobot pecahan; entropy([]) = 0."""
    y, weights = _labels_weights(y, sample_weight)
    return float(_entropy_counts(np.bincount(y, weights=weights, minlength=2)))


def _missing(column):
    """Tandai nilai None dan NaN sebagai nilai yang hilang."""
    if column.dtype.kind in "fci":
        return np.isnan(column)
    return np.fromiter((v is None or v != v for v in column),
                       dtype=bool, count=len(column))


def _partition_scores(counts, missing_weight):
    """Hitung gain terkoreksi dan split information dengan memperhitungkan bobot missing."""
    counts = np.asarray(counts, dtype=float)
    masses = counts.sum(axis=1)
    known = masses.sum()
    total = known + missing_weight
    if known <= 0 or total <= 0:
        return 0.0, 0.0
    gain_known = (_entropy_counts(counts.sum(axis=0))
                  - np.dot(masses / known, _entropy_counts(counts)))
    gain = max(0.0, float(known / total * gain_known))
    split_info = float(_entropy_counts(np.append(masses, missing_weight)))
    return gain, split_info


def information_gain(y, branches, sample_weight=None):
    """Gain split; None/NaN dalam branches menandakan atribut tidak diketahui."""
    y, weights = _labels_weights(y, sample_weight)
    branches = np.asarray(branches)
    if branches.shape != y.shape:
        raise ValueError("branches harus satu dimensi dengan panjang sama dengan y.")
    missing = _missing(branches)
    values, inverse = np.unique(branches[~missing], return_inverse=True)
    counts = np.zeros((len(values), 2))
    np.add.at(counts, (inverse, y[~missing]), weights[~missing])
    return _partition_scores(counts, weights[missing].sum())[0]


def gain_ratio(y, branches, sample_weight=None):
    """Information gain dibagi split information; split konstan bernilai nol."""
    y, weights = _labels_weights(y, sample_weight)
    branches = np.asarray(branches)
    if branches.shape != y.shape:
        raise ValueError("branches harus satu dimensi dengan panjang sama dengan y.")
    missing = _missing(branches)
    values, inverse = np.unique(branches[~missing], return_inverse=True)
    counts = np.zeros((len(values), 2))
    np.add.at(counts, (inverse, y[~missing]), weights[~missing])
    gain, split_info = _partition_scores(counts, weights[missing].sum())
    return gain / split_info if split_info > 1e-12 else 0.0


@dataclass
class TreeNode:
    """Statistik kelas berbobot dan aturan split; children kosong berarti daun."""

    class_counts: np.ndarray
    n_samples: int
    depth: int
    feature: int | None = None
    threshold: float | None = None
    categorical: bool = False
    children: dict = field(default_factory=dict)
    branch_probabilities: dict = field(default_factory=dict)
    information_gain: float = 0.0
    gain_ratio: float = 0.0

    def is_leaf(self):
        """Periksa apakah simpul tidak memiliki anak."""
        return not self.children

    @property
    def probability(self):
        """Hitung peluang kelas 1 dari jumlah kelas berbobot."""
        return float(self.class_counts[1] / self.class_counts.sum())


@dataclass
class _Split:
    feature: int
    threshold: float | None
    categories: tuple | None
    gain: float
    ratio: float
    probabilities: np.ndarray


class DecisionTree:
    """Pohon berbasis C4.5 mengacu pada salindia kuliah Decision Tree.pdf dan DTL.pdf."""

    def __init__(self, categorical_features=(), max_depth=20,
                 min_samples_split=2, min_samples_leaf=1.0, min_gain=1e-12):
        """Simpan indeks fitur kategorikal dan parameter penghentian pertumbuhan pohon."""
        self.categorical_features = tuple(categorical_features)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain

    def _validate_parameters(self, n_features):
        """Validasi parameter pertumbuhan dan indeks fitur kategorikal."""
        if (self.max_depth is not None and
                (isinstance(self.max_depth, bool)
                 or not isinstance(self.max_depth, Integral) or self.max_depth < 0)):
            raise ValueError("max_depth harus integer nonnegatif atau None.")
        if (isinstance(self.min_samples_split, bool)
                or not isinstance(self.min_samples_split, Integral)
                or self.min_samples_split < 2):
            raise ValueError("min_samples_split harus integer >= 2.")
        for name, value, positive in [("min_samples_leaf", self.min_samples_leaf, True),
                                      ("min_gain", self.min_gain, False)]:
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value < 0 or (positive and value == 0)):
                raise ValueError(f"{name} harus finite dan {'positif' if positive else 'nonnegatif'}.")
        features = self.categorical_features
        if any(isinstance(f, bool) or not isinstance(f, Integral)
               or f < 0 or f >= n_features for f in features):
            raise ValueError("Indeks fitur kategorikal di luar rentang atau bukan integer.")
        if len(set(features)) != len(features):
            raise ValueError("Indeks fitur kategorikal tidak boleh duplikat.")

    @staticmethod
    def _validate_X(X, categorical_features, n_features=None):
        """Validasi dan salin fitur numerik atau kategorikal serta seragamkan nilai missing."""
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X harus array dua dimensi dengan setidaknya satu fitur.")
        if np.iscomplexobj(X):
            raise ValueError("X tidak boleh berisi bilangan kompleks.")
        if n_features is not None and X.shape[1] != n_features:
            raise ValueError("Jumlah fitur berbeda dengan data training.")
        result = np.array(X, dtype=object if categorical_features else float, copy=True)
        for feature in range(X.shape[1]):
            column = result[:, feature]
            missing = _missing(column)
            known = column[~missing]
            if feature in categorical_features:
                strings = all(isinstance(v, (str, np.str_)) for v in known)
                numbers = all(isinstance(v, Real) and np.isfinite(v) for v in known)
                if not (strings or numbers):
                    raise ValueError("Kategori per kolom harus seragam: string atau angka finite.")
            else:
                try:
                    values = np.asarray(known, dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Kolom nonkategorikal harus numerik.") from exc
                if not np.isfinite(values).all():
                    raise ValueError("Fitur numerik tidak boleh berisi infinity.")
                column[~missing] = values
            column[missing] = np.nan
        return result

    def _best_numeric(self, x, y, weights, feature):
        """Cari ambang numerik dengan gain ratio terbaik menggunakan hitungan kelas kumulatif."""
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
        split_counts = np.column_stack((masses, np.full(len(masses), missing_w)))
        split_info = _entropy_counts(split_counts)
        ratios = np.divide(gain, split_info, out=np.zeros_like(gain), where=split_info > 1e-12)
        valid &= (gain > self.min_gain) & (split_info > 1e-12)
        if not valid.any():
            return None
        best = int(np.argmax(np.where(valid, ratios, -np.inf)))
        a, b = values[positions[best]], values[positions[best] + 1]
        threshold = a / 2.0 + b / 2.0
        if not a <= threshold < b:
            threshold = a
        return _Split(feature, float(threshold), None, float(gain[best]),
                      float(ratios[best]), probabilities[best])

    def _best_categorical(self, x, y, weights, feature):
        """Evaluasi split per kategori berdasarkan gain ratio dan batas bobot daun."""
        known = ~_missing(x)
        categories, inverse = np.unique(x[known], return_inverse=True)
        if len(categories) < 2:
            return None
        counts = np.zeros((len(categories), 2))
        np.add.at(counts, (inverse, y[known]), weights[known])
        masses = counts.sum(axis=1)
        probabilities = masses / masses.sum()
        if (probabilities * weights.sum() < self.min_samples_leaf - 1e-12).any():
            return None
        gain, split_info = _partition_scores(counts, weights[~known].sum())
        if gain <= self.min_gain or split_info <= 1e-12:
            return None
        return _Split(feature, None, tuple(categories), gain, gain / split_info, probabilities)

    @staticmethod
    def _branches(x, split):
        """Tentukan indeks cabang setiap sampel dengan penanda -1 untuk nilai missing."""
        branches = np.full(len(x), -1, dtype=int)
        known = ~_missing(x)
        if split.categories is None:
            branches[known] = (np.asarray(x[known], dtype=float) > split.threshold).astype(int)
        else:
            for branch, category in enumerate(split.categories):
                branches[known & (x == category)] = branch
        return branches

    def fit(self, X, y):
        """Latih pohon dari data training dengan bobot pecahan tanpa memodifikasi input."""
        raw = np.asarray(X)
        if raw.ndim != 2:
            raise ValueError("X harus array dua dimensi.")
        self._validate_parameters(raw.shape[1])
        categorical = frozenset(self.categorical_features)
        X = self._validate_X(X, categorical)
        y, weights = _labels_weights(y)
        if len(y) == 0 or len(y) != len(X):
            raise ValueError("X dan y harus tidak kosong dan memiliki jumlah baris sama.")
        root = TreeNode(np.bincount(y, minlength=2).astype(float), len(y), 0)
        n_nodes, depth = 1, 0
        stack = [(root, np.arange(len(y)), weights, categorical)]
        while stack:
            node, indices, w, available_categories = stack.pop()
            if (np.count_nonzero(node.class_counts) < 2
                    or len(indices) < self.min_samples_split
                    or w.sum() < 2 * self.min_samples_leaf
                    or (self.max_depth is not None and node.depth >= self.max_depth)):
                continue
            candidates = []
            for feature in range(X.shape[1]):
                if feature in categorical and feature not in available_categories:
                    continue
                search = self._best_categorical if feature in categorical else self._best_numeric
                candidate = search(X[indices, feature], y[indices], w, feature)
                if candidate is not None:
                    candidates.append(candidate)
            if not candidates:
                continue
            average_gain = np.mean([s.gain for s in candidates])
            eligible = [s for s in candidates if s.gain >= average_gain - 1e-12]
            split = max(eligible, key=lambda s: s.ratio)
            branches = self._branches(X[indices, split.feature], split)
            missing = branches == -1
            node.feature, node.threshold = split.feature, split.threshold
            node.categorical = split.categories is not None
            node.information_gain, node.gain_ratio = split.gain, split.ratio
            keys = split.categories if node.categorical else (0, 1)
            next_categories = available_categories - {split.feature} if node.categorical else available_categories
            for branch, key in enumerate(keys):
                probability = float(split.probabilities[branch])
                mask = (branches == branch) | missing
                child_indices = indices[mask]
                child_w = w[mask] * np.where(missing[mask], probability, 1.0)
                counts = np.bincount(y[child_indices], weights=child_w, minlength=2)
                child = TreeNode(counts, len(child_indices), node.depth + 1)
                node.children[key] = child
                node.branch_probabilities[key] = probability
                n_nodes += 1
                depth = max(depth, child.depth)
                stack.append((child, child_indices, child_w, next_categories))
        self.root = root
        self.n_features_in_ = X.shape[1]
        self.categorical_features_ = categorical
        self.n_nodes_, self.depth_ = n_nodes, depth
        self.classes_ = np.array([0, 1])
        return self

    def _check_fitted(self):
        """Pastikan model sudah dilatih sebelum digunakan."""
        if not hasattr(self, "root"):
            raise RuntimeError("Model belum dilatih; panggil fit terlebih dahulu.")

    def predict_proba(self, X):
        """Peluang kelas 1; missing/kategori baru menggabungkan semua cabang."""
        self._check_fitted()
        X = self._validate_X(X, self.categorical_features_, self.n_features_in_)
        result = np.zeros(len(X))
        stack = [(self.root, np.arange(len(X)), np.ones(len(X)))]
        while stack:
            node, indices, w = stack.pop()
            if not len(indices):
                continue
            if node.is_leaf():
                result[indices] += w * node.probability
                continue
            x = X[indices, node.feature]
            known = ~_missing(x)
            if node.categorical:
                matched = np.zeros(len(x), dtype=bool)
                masks = {}
                for key in node.children:
                    masks[key] = known & (x == key)
                    matched |= masks[key]
            else:
                left = np.zeros(len(x), dtype=bool)
                left[known] = np.asarray(x[known], dtype=float) <= node.threshold
                masks = {0: known & left, 1: known & ~left}
                matched = known
            unknown = ~matched
            for key, child in node.children.items():
                mask = masks[key] | unknown
                child_w = w[mask] * np.where(unknown[mask], node.branch_probabilities[key], 1.0)
                stack.append((child, indices[mask], child_w))
        return np.clip(result, 0.0, 1.0)

    def predict(self, X, threshold=0.5):
        """Kelas 1 jika probabilitas >= threshold, termasuk saat tepat sama."""
        if (not isinstance(threshold, Real) or not np.isfinite(threshold)
                or not 0 <= threshold <= 1):
            raise ValueError("threshold harus angka finite dalam [0, 1].")
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def save(self, path):
        """Simpan model terlatih dengan pickle dari standard library."""
        self._check_fitted()
        with open(path, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """Muat hanya berkas pickle tepercaya; pickle dapat mengeksekusi kode."""
        with open(path, "rb") as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise ValueError("Berkas tidak berisi model DecisionTree.")
        model._check_fitted()
        return model

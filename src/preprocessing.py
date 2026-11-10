"""Preprocessing: split, cleaning, langkah persiapan, dan pipeline. Semua `fit` hanya pada train."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

TARGET = "is_canceled"
ID_COL = "id"
CATEGORICAL = ["meal_package", "room_category", "booking_channel"]


def feature_cols(df):
    return [c for c in df.columns if c not in (ID_COL, TARGET)]


def split_train_val(df, val_size=0.2, seed=42):
    """Split train.csv berstratifikasi pada target."""
    return train_test_split(df, test_size=val_size, random_state=seed, stratify=df[TARGET])


def drop_duplicates_and_invalid(df):
    """Buang duplikat dan baris invalid (hanya untuk train). Kembalikan (df_bersih, laporan)."""
    feats = feature_cols(df)
    report = {"rows_in": len(df)}

    # grup duplikat dengan label bertentangan dibuang seluruhnya, grup lain disisakan satu baris
    conflict = df.groupby(feats, dropna=False)[TARGET].transform("nunique") > 1
    report["conflicting_duplicates"] = int(conflict.sum())
    df = df[~conflict]
    dup = df.duplicated(subset=feats)
    report["duplicates"] = int(dup.sum())
    df = df[~dup]

    # malam = 0, tarif = 0, dan dewasa = 0 sengaja dipertahankan (alasan di notebook)
    num = df[feats].select_dtypes("number")
    invalid = (df["num_adults"].eq(0) & df["num_children"].eq(0)) | (num < 0).any(axis=1)
    report["invalid"] = int(invalid.sum())
    df = df[~invalid]

    report["rows_out"] = len(df)
    return df, report


def construct_features(df):
    """Tambah total_nights, total_guests, nightly_rate_per_guest."""
    df = df.copy()
    df["total_nights"] = df["weekend_nights"] + df["weekday_nights"]
    df["total_guests"] = df["num_adults"] + df["num_children"]
    df["nightly_rate_per_guest"] = df["nightly_rate_idr"] / df["total_guests"].clip(lower=1)
    return df


class Imputer:
    """Isi NaN: median (numerik), modus (kategorikal). Kolom di indicator_cols diberi <nama>_missing."""

    def __init__(self, indicator_cols=("special_requests",)):
        self.indicator_cols = list(indicator_cols)

    def fit(self, df):
        cols = feature_cols(df)
        num = [c for c in cols if c not in CATEGORICAL]
        self.fill_ = {**df[num].median().to_dict(),
                      **{c: df[c].mode()[0] for c in cols if c in CATEGORICAL}}
        return self

    def transform(self, df):
        df = df.copy()
        for c in self.indicator_cols:
            df[f"{c}_missing"] = df[c].isna().astype(int)
        return df.fillna(self.fill_)


class Clipper:
    """Potong kolom kontinu ke kuantil [lower, upper] train."""

    # kolom hitungan (mis. prior_cancellations) tidak dipotong: kuantil 99%-nya 0, sinyalnya akan hilang
    def __init__(self, cols=("days_before_arrival", "nightly_rate_idr", "nightly_rate_per_guest"),
                 lower=0.01, upper=0.99):
        self.cols, self.lower, self.upper = list(cols), lower, upper

    def fit(self, df):
        self.bounds_ = df[self.cols].quantile([self.lower, self.upper])
        return self

    def transform(self, df):
        df = df.copy()
        for c in self.bounds_.columns:
            df[c] = df[c].clip(*self.bounds_[c])
        return df


class CategoricalEncoder:
    """One-hot dengan kategori dari train. Kategori tak dikenal menjadi semua 0."""

    def fit(self, df):
        self.categories_ = {c: sorted(df[c].dropna().unique()) for c in CATEGORICAL}
        return self

    def transform(self, df):
        df = df.copy()
        for c, cats in self.categories_.items():
            for v in cats:
                df[f"{c}={v}"] = (df[c] == v).astype(int)
        return df.drop(columns=list(self.categories_))


class Scaler:
    """Standarisasi ("standard") atau min-max ("minmax") dengan statistik train."""

    def __init__(self, method="standard"):
        self.method = method

    def fit(self, df):
        X = df[feature_cols(df)]
        if self.method == "standard":
            self.shift_, scale = X.mean(), X.std(ddof=0)
        else:
            self.shift_, scale = X.min(), X.max() - X.min()
        self.scale_ = scale.replace(0, 1)  # kolom konstan: hindari bagi nol
        return self

    def transform(self, df):
        df = df.copy()
        cols = list(self.shift_.index)
        df[cols] = (df[cols] - self.shift_) / self.scale_
        return df


class FeatureSelector:
    """Pilih k fitur dengan mutual information tertinggi terhadap target. Skor di `scores_`."""

    def __init__(self, k=None, min_score=0.0, seed=42):
        self.k, self.min_score, self.seed = k, min_score, seed

    def fit(self, df):
        cols = feature_cols(df)
        mi = mutual_info_classif(df[cols], df[TARGET], random_state=self.seed)
        self.scores_ = pd.Series(mi, index=cols).sort_values(ascending=False)
        keep = self.scores_[self.scores_ >= self.min_score]
        self.selected_ = list(keep.index[:self.k] if self.k else keep.index)
        return self

    def transform(self, df):
        return df[[c for c in df.columns if c in self.selected_ or c in (ID_COL, TARGET)]]


class DimReducer:
    """PCA. n_components < 1 berarti proporsi varians yang dipertahankan."""

    def __init__(self, n_components=0.95, seed=42):
        self.n_components, self.seed = n_components, seed

    def fit(self, df):
        self.pca_ = PCA(self.n_components, random_state=self.seed).fit(df[feature_cols(df)])
        return self

    def transform(self, df):
        Z = self.pca_.transform(df[feature_cols(df)])
        out = pd.DataFrame(Z, index=df.index, columns=[f"pc{i + 1}" for i in range(Z.shape[1])])
        return out.join(df[[c for c in (ID_COL, TARGET) if c in df.columns]])


def handle_imbalance(X, y, seed=42):
    """Random oversampling kelas minoritas sampai seimbang. Hanya untuk train."""
    X, y = np.asarray(X), np.asarray(y)
    counts = np.bincount(y)
    minority = counts.argmin()
    extra = np.random.default_rng(seed).choice(np.flatnonzero(y == minority), counts.max() - counts.min())
    idx = np.r_[np.arange(len(y)), extra]
    return X[idx], y[idx]


class PreparationPipeline:
    """Jalankan langkah berurutan. transform mengembalikan (X, y); y = None bila tidak ada target."""

    def __init__(self, steps):
        self.steps = steps

    def _run(self, df, fit):
        df = df.drop(columns=[ID_COL], errors="ignore")
        for step in self.steps:
            if callable(step):  # fungsi tanpa state, mis. construct_features
                df = step(df)
            else:
                df = (step.fit(df) if fit else step).transform(df)
        return df

    def fit(self, df):
        self.columns_ = feature_cols(self._run(df, fit=True))
        return self

    def transform(self, df):
        out = self._run(df, fit=False)
        y = out[TARGET].to_numpy().astype(int) if TARGET in out else None
        return out[self.columns_], y

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def numeric_pipeline(select_k=None, pca=None):
    """Tampilan numerik (KNN, LR, SVM): tanpa NaN, one-hot, terskala."""
    steps = [construct_features, Imputer(), Clipper(), CategoricalEncoder(), Scaler()]
    if select_k:
        steps.append(FeatureSelector(k=select_k))
    if pca:
        steps.append(DimReducer(pca))
    return PreparationPipeline(steps)


def tree_pipeline(impute=False):
    """Tampilan pohon (DTL, NB): kategori tetap teks, NaN dibiarkan kecuali impute=True."""
    return PreparationPipeline([construct_features] + ([Imputer()] if impute else []))


def prepare(train_df, val_df, view="numeric", **kw):
    """Bersihkan train, fit pipeline, transform keduanya. Untuk evaluation.cross_validate."""
    train_df, _ = drop_duplicates_and_invalid(train_df)
    pipe = (numeric_pipeline if view == "numeric" else tree_pipeline)(**kw).fit(train_df)
    return (*pipe.transform(train_df), *pipe.transform(val_df))


def make_views(train_df, *others):
    """Fit kedua pipeline pada train, transform semua df. Kembalikan {"numeric": [(X, y), ...], "tree": [...]}."""
    out = {}
    for name, pipe in [("numeric", numeric_pipeline()), ("tree", tree_pipeline())]:
        pipe.fit(train_df)
        out[name] = [pipe.transform(d) for d in (train_df, *others)]
    return out

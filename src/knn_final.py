"""Model akhir (submission KNN v3): KNN manhattan k=18 berbobot 1/d^1.5 di ruang fitur berbobot.

Langkah di atas pipeline numerik standar:
1. Fitur tanggal: hari ke- sejak 2017-01-01, hari dalam minggu/tahun, tanggal pesan, harga total.
2. Bobot kolom = (MI / rata-rata MI)^0.5 dikali pengali per atribut (FEATURE_MULT). MI dihitung pada train saja.
   Pengali dicari dengan coordinate ascent pada fold 0-2 (fold 3-4 holdout), validasi stratified 5-fold seed 42.
3. days_before_arrival yang hilang (dan booking_ord turunannya) diisi rata-rata 10 tetangga train yang
   lead-nya teramati, jarak memakai kolom lain.
CV macro F1 (5-fold, threshold 0.5): 0.8521.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from knn import KNN
from preprocessing import (ID_COL, TARGET, CategoricalEncoder, Clipper, Imputer, PreparationPipeline, Scaler,
                           construct_features, drop_duplicates_and_invalid)

# atribut yang tidak tercantum memakai pengali 1; kolom one-hot ikut atribut asalnya (nama sebelum "=")
FEATURE_MULT = {"arrival_doy": 0.0, "arrival_weekday": 0.0, "nightly_rate_per_guest": 0.0,
                "special_requests_missing": 0.0, "total_price": 0.0, "arrival_month": 0.5, "total_nights": 0.5,
                "booking_ord": 1.5, "needs_parking": 1.5, "booking_channel": 2.5}
LEAD_COLS = ("days_before_arrival", "booking_ord")


def date_features(df):
    """Tanggal kedatangan -> hari ke-, hari dalam minggu/tahun; tanggal pesan = kedatangan - days_before_arrival."""
    df = df.copy()
    arr = pd.to_datetime(dict(year=df["arrival_year"], month=df["arrival_month"], day=df["arrival_day"]),
                         errors="coerce")  # tanggal tak valid (mis. 29 Feb) -> NaT
    df["arrival_ord"] = (arr - pd.Timestamp("2017-01-01")).dt.days
    df["arrival_weekday"] = arr.dt.weekday
    df["arrival_doy"] = arr.dt.dayofyear
    df["booking_ord"] = df["arrival_ord"] - df["days_before_arrival"]
    df["total_price"] = df["nightly_rate_idr"] * (df["weekend_nights"] + df["weekday_nights"])
    return df


class PowKNN(KNN):
    """KNN dengan bobot tetangga 1/d^q (q=1 sama dengan weights="distance")."""

    def __init__(self, q=1.5, **kw):
        super().__init__(**kw)
        self.q = q

    def predict_proba(self, X):
        dist, idx = self.kneighbors(X)
        w = 1 / np.maximum(dist, 1e-12) ** self.q
        return (w * self.y_[idx]).sum(1) / w.sum(1)


def prepare_final(train_df, other_df):
    """Fit pada train_df (setelah buang duplikat/invalid), transform keduanya. Kembalikan X_tr, y_tr, X_other."""
    train_df, _ = drop_duplicates_and_invalid(train_df)
    pipe = PreparationPipeline([construct_features, date_features, Imputer(), Clipper(), CategoricalEncoder(),
                                Scaler()]).fit(train_df)
    X_tr, y_tr = pipe.transform(train_df)
    X_ot, _ = pipe.transform(other_df)
    cols = list(pipe.columns_)
    mi = mutual_info_classif(X_tr.to_numpy(), y_tr, random_state=42)
    w = (mi / mi.mean()) ** 0.5 * np.array([FEATURE_MULT.get(c.split("=")[0], 1.0) for c in cols])
    X_tr, X_ot = X_tr.to_numpy() * w, X_ot.to_numpy() * w

    # imputasi lead dengan KNN (k=10) dari baris train yang lead-nya teramati
    lead = [cols.index(c) for c in LEAD_COLS]
    other = [j for j in range(len(cols)) if j not in lead]
    miss_tr = train_df["days_before_arrival"].isna().to_numpy()
    miss_ot = other_df["days_before_arrival"].isna().to_numpy()
    ref = X_tr[~miss_tr]
    finder = KNN(k=10, metric="manhattan").fit(ref[:, other], np.zeros(len(ref)))
    for X, miss in ((X_tr, miss_tr), (X_ot, miss_ot)):
        if miss.any():
            _, idx = finder.kneighbors(X[miss][:, other])
            for j in lead:
                X[np.flatnonzero(miss), j] = ref[:, j][idx].mean(1)
    return X_tr, y_tr, X_ot


def final_model():
    return PowKNN(k=18, q=1.5, metric="manhattan")


def make_final_submission(train_df, test_df, path):
    """Tulis CSV `id,is_canceled` dari model akhir. Kembalikan DataFrame yang ditulis."""
    X_tr, y_tr, X_te = prepare_final(train_df, test_df)
    pred = final_model().fit(X_tr, y_tr).predict(X_te, threshold=0.5)
    sub = pd.DataFrame({ID_COL: test_df[ID_COL].to_numpy(), TARGET: pred})
    assert len(sub) == len(test_df) and sub[ID_COL].is_unique
    sub.to_csv(path, index=False)
    return sub

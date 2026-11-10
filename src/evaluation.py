"""Modul evaluasi bersama. Semua fungsi menerima y (0/1) dan y_pred (0/1) atau skor peluang."""
import numpy as np
import pandas as pd


def confusion_matrix(y, y_pred):
    """[[TN, FP], [FN, TP]], urutan sama dengan sklearn.metrics.confusion_matrix."""
    y, y_pred = np.asarray(y).astype(int), np.asarray(y_pred).astype(int)
    return np.array([[np.sum((y == 0) & (y_pred == 0)), np.sum((y == 0) & (y_pred == 1))],
                     [np.sum((y == 1) & (y_pred == 0)), np.sum((y == 1) & (y_pred == 1))]])


def accuracy(y, y_pred):
    return float(np.mean(np.asarray(y) == np.asarray(y_pred)))


def precision(y, y_pred):
    (_, fp), (_, tp) = confusion_matrix(y, y_pred)
    return float(tp / (tp + fp)) if tp + fp else 0.0


def recall(y, y_pred):
    (_, _), (fn, tp) = confusion_matrix(y, y_pred)
    return float(tp / (tp + fn)) if tp + fn else 0.0


def f1(y, y_pred):
    p, r = precision(y, y_pred), recall(y, y_pred)
    return 2 * p * r / (p + r) if p + r else 0.0


def roc_auc(y, score):
    """AUC = peluang skor positif > skor negatif (statistik Mann-Whitney). Skor kembar diberi rata-rata peringkat."""
    y, score = np.asarray(y).astype(int), np.asarray(score, dtype=float)
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s))
    # peringkat 1..n, lalu tiap kelompok skor sama diberi rata-rata peringkatnya
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    ends = np.r_[starts[1:], len(s)]
    for a, b in zip(starts, ends):
        ranks[a:b] = (a + 1 + b) / 2
    r = np.empty(len(s))
    r[order] = ranks
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def scores(y, proba, threshold=0.5):
    """Semua metrik dalam satu dict, dari peluang kelas 1."""
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    return {"accuracy": accuracy(y, y_pred), "precision": precision(y, y_pred),
            "recall": recall(y, y_pred), "f1": f1(y, y_pred), "roc_auc": roc_auc(y, proba)}


def stratified_kfold(y, k=5, seed=42):
    """Daftar (idx_train, idx_val). Tiap kelas diacak lalu dibagi rata ke k fold, jadi proporsi kelas tiap fold sama."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        fold_of[idx] = np.arange(len(idx)) % k
    return [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)) for f in range(k)]


def cross_validate(model_fn, prepare, df, k=5, seed=42, target="is_canceled"):
    """Stratified k-fold. `prepare(train_df, val_df) -> (X_tr, y_tr, X_va, y_va)` dipanggil per fold,
    jadi preprocessing di-fit hanya pada bagian train fold itu (tanpa leakage). Kembalikan metrik per fold."""
    rows = []
    for f, (tr, va) in enumerate(stratified_kfold(df[target].to_numpy(), k, seed)):
        X_tr, y_tr, X_va, y_va = prepare(df.iloc[tr], df.iloc[va])
        proba = model_fn().fit(X_tr, y_tr).predict_proba(X_va)
        rows.append({"fold": f, **scores(y_va, proba)})
    out = pd.DataFrame(rows).set_index("fold")
    out.loc["mean"] = out.mean()
    return out


def compare_with_sklearn(pairs, X_tr, y_tr, X_va, y_va):
    """pairs: {nama: (model_scratch, model_sklearn)}, keduanya belum di-fit. Satu baris per (model, versi)."""
    rows = []
    for name, (scratch, sk) in pairs.items():
        for impl, m in [("scratch", scratch), ("sklearn", sk)]:
            p = m.fit(X_tr, y_tr).predict_proba(X_va)
            p = p[:, 1] if p.ndim == 2 else p  # sklearn mengembalikan (m, 2), scratch (m,)
            rows.append({"model": name, "impl": impl, **scores(y_va, p)})
    return pd.DataFrame(rows).set_index(["model", "impl"])


def error_analysis(df_val, y, y_pred, cols=None):
    """Bandingkan rata-rata fitur (numerik) pada TP/TN/FP/FN. df_val berisi fitur mentah baris validasi."""
    y, y_pred = np.asarray(y), np.asarray(y_pred)
    kind = np.select([(y == 1) & (y_pred == 1), (y == 0) & (y_pred == 0), (y == 0) & (y_pred == 1)],
                     ["TP", "TN", "FP"], "FN")
    num = df_val[cols] if cols else df_val.select_dtypes("number")
    out = num.groupby(kind).mean().T
    out.loc["n_rows"] = pd.Series(kind).value_counts()
    return out

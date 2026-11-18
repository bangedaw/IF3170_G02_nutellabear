"""Submission Kaggle: fit ulang pipeline dan model pada seluruh train.csv, test.csv hanya di-transform."""
import pandas as pd

from preprocessing import ID_COL, TARGET, drop_duplicates_and_invalid


def make_submission(model, pipeline, train_df, test_df, path, threshold=0.5):
    """Tulis CSV `id,is_canceled` berformat sample_submission.csv. Kembalikan DataFrame yang ditulis."""
    train_df, _ = drop_duplicates_and_invalid(train_df)
    pipeline.fit(train_df)
    X_tr, y_tr = pipeline.transform(train_df)
    X_te, _ = pipeline.transform(test_df)
    pred = model.fit(X_tr, y_tr).predict(X_te, threshold)
    sub = pd.DataFrame({ID_COL: test_df[ID_COL].to_numpy(), TARGET: pred})
    assert len(sub) == len(test_df) and sub[ID_COL].is_unique
    sub.to_csv(path, index=False)
    return sub


def check_reproducible(path_a, path_b):
    """True bila dua file submission berisi id dan prediksi yang sama (urutan baris bebas)."""
    a = pd.read_csv(path_a, dtype={ID_COL: str}).sort_values(ID_COL).reset_index(drop=True)
    b = pd.read_csv(path_b, dtype={ID_COL: str}).sort_values(ID_COL).reset_index(drop=True)
    return a.equals(b)

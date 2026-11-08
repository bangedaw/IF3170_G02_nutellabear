import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TARGET = "is_canceled"
ID_COL = "id"
CATEGORICAL = ["meal_package", "room_category", "booking_channel"]

def find_invalid_rows(df):
    nights = df["weekend_nights"] + df["weekday_nights"]
    return {
        "zero_nights": nights == 0,
        "no_guests": (df["num_adults"] == 0) & (df["num_children"] == 0),
    }

def iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

def describe_data(df):
    feats = df.drop(columns=[c for c in (ID_COL,) if c in df.columns])
    info = pd.DataFrame({
        "dtype": feats.dtypes.astype(str),
        "n_missing": feats.isna().sum(),
        "n_unique": feats.nunique(),
    })
    print(f"Ukuran: {df.shape}")
    print(info.to_string())
    print(feats.describe().T.round(2).to_string())
    for c in CATEGORICAL:
        print(f"\n{c}")
        print(feats[c].value_counts(dropna=False).to_string())
    return info

def verify_quality(df, name="train"):
    feats = df.drop(columns=[c for c in (ID_COL, TARGET) if c in df.columns])
    n = len(df)

    miss = df.isna().sum()
    miss = miss[miss > 0]
    report = {
        "n_rows": n,
        "missing": pd.DataFrame({"n": miss, "pct": (miss / n * 100).round(2)}),
        "rows_with_any_missing": int(df.isna().any(axis=1).sum()),
        "duplicates_extra_copies": int(feats.duplicated().sum()),
    }

    dup_mask = feats.duplicated(keep=False)
    if TARGET in df.columns and dup_mask.any():
        grp = df[dup_mask].groupby(list(feats.columns), dropna=False)[TARGET].nunique()
        report["duplicate_groups_conflicting_label"] = int((grp > 1).sum())

    report["invalid"] = {k: int(v.sum()) for k, v in find_invalid_rows(df).items()}

    num = feats.select_dtypes("number")
    report["iqr_outliers"] = {
        c: iqr_outliers(num[c].dropna()) for c in num.columns if num[c].nunique() > 5
    }

    if TARGET in df.columns:
        report["class_share"] = df[TARGET].value_counts(normalize=True).round(4).to_dict()

    print(f"=== Kualitas data: {name} ({n} baris) ===")
    print(report["missing"].to_string())
    print(f"Baris dengan >=1 missing: {report['rows_with_any_missing']}")
    print(
        f"Duplikat tanpa id: {report['duplicates_extra_copies']} salinan berlebih, "
        f"grup label konflik: {report.get('duplicate_groups_conflicting_label', 'n/a')}"
    )
    print(f"Nilai tidak valid: {report['invalid']}")
    print(f"Outlier IQR: {report['iqr_outliers']}")
    if "class_share" in report:
        print(f"Proporsi kelas: {report['class_share']}")
    return report

def finish(fig, name, save_dir):
    fig.tight_layout()
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / f"{name}.png", dpi=150)
    plt.show()

def explore_data(df, save_dir=None):
    num_cols = [c for c in df.select_dtypes("number").columns if c != TARGET]

    fig, ax = plt.subplots(figsize=(4, 4))
    df[TARGET].value_counts(normalize=True).sort_index().plot.bar(ax=ax)
    ax.set_title("Proporsi kelas")
    finish(fig, "target_distribution", save_dir)

    fig, axes = plt.subplots(1, len(CATEGORICAL), figsize=(5 * len(CATEGORICAL), 4))
    for ax, c in zip(axes, CATEGORICAL):
        df.groupby(c, dropna=False)[TARGET].mean().sort_values().plot.barh(ax=ax)
        ax.axvline(df[TARGET].mean(), color="red", linestyle="--")
        ax.set_title(f"Rasio batal per {c}")
    finish(fig, "cancel_rate_categorical", save_dir)

    ncols = 4
    nrows = int(np.ceil(len(num_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    for ax, c in zip(axes.ravel(), num_cols):
        ax.hist(df[c].dropna(), bins=30)
        ax.set_title(c)
    for ax in axes.ravel()[len(num_cols):]:
        ax.axis("off")
    finish(fig, "numeric_distributions", save_dir)

    corr = df[num_cols + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax)
    ax.set_title("Korelasi")
    finish(fig, "correlation", save_dir)

    return corr[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
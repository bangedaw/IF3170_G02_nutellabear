"""Tes submission: `python -m pytest tests/test_submission.py` dari root repo."""
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from knn import KNN  # noqa: E402
from preprocessing import numeric_pipeline  # noqa: E402
from submission import check_reproducible, make_submission  # noqa: E402

DATA = ROOT / "T2-IF3170-2026-2027"
train = pd.read_csv(DATA / "train.csv", dtype={"id": str}).head(3000)  # subset agar tes cepat
test = pd.read_csv(DATA / "test.csv", dtype={"id": str})
sample = pd.read_csv(DATA / "sample_submission.csv", dtype={"id": str})


def test_format_and_reproducible():
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a.csv", Path(d) / "b.csv"
        make_submission(KNN(15, weights="distance"), numeric_pipeline(), train, test, a)
        make_submission(KNN(15, weights="distance"), numeric_pipeline(), train, test, b)
        sub = pd.read_csv(a, dtype={"id": str})
        assert list(sub.columns) == list(sample.columns)
        assert len(sub) == 6844 and set(sub.id) == set(sample.id)
        assert set(sub.is_canceled.unique()) <= {0, 1}
        assert check_reproducible(a, b)

        sub.loc[0, "is_canceled"] = 1 - sub.loc[0, "is_canceled"]  # satu prediksi beda harus terdeteksi
        sub.to_csv(b, index=False)
        assert not check_reproducible(a, b)

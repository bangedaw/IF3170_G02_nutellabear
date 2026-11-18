import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib

matplotlib.use("Agg")
from eda import compare_train_test, ks_statistic


def test_ks_known_values():
    assert ks_statistic([1, 2, 3], [1, 2, 3]) == 0.0
    assert ks_statistic([1, 2, 3], [10, 11, 12]) == 1.0
    assert np.isclose(ks_statistic([1, 3], [2, 4]), 0.5)


def test_compare_flags_shift_and_new_category():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"id": "a", "x": rng.normal(0, 1, 500), "c": ["u", "v"] * 250, "is_canceled": 0})
    test = pd.DataFrame({"id": "b", "x": rng.normal(2, 1, 200), "c": ["u", "w"] * 100})
    out = compare_train_test(train, test)
    assert set(out.index) == {"x", "c"}
    assert out.loc["x", "distance"] > 0.5 and out.loc["x", "mean_shift_std"] > 1.5
    assert out.loc["c", "new_in_test"] == ["w"]

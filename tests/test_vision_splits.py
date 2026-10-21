"""El split de imágenes no debe mezclar grupos ni copias exactas entre conjuntos."""

import pandas as pd
import pytest

from src.vision.splits import SPLITS_PATH, check_no_leakage


@pytest.mark.skipif(not SPLITS_PATH.exists(), reason="ejecutar python -m src.vision.splits")
def test_sin_fuga_de_datos():
    splits = pd.read_csv(SPLITS_PATH)
    check_no_leakage(splits)
    assert set(splits["split"]) == {"train", "val", "test"}
    assert splits.loc[splits["dataset"] == "bracol", "split"].eq("test").sum() > 0

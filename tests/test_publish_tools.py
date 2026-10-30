"""Revisión previa a publicar y figuras del informe, con datos sintéticos."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.common import figures, publish_check


def test_scan_text_finds_secrets_and_local_paths():
    # se arma por partes para que esta misma prueba no dispare la revisión
    text = "api_" + 'key = "abcdef123456"\nruta = "C:' + "\\Users\\" + 'alguien\\proyecto"\n'
    kinds = {k for k, _ in publish_check.scan_text(text)}
    assert "posible credencial en texto" in kinds and "ruta local" in kinds
    assert publish_check.scan_text("texto normal sin nada raro") == []


@pytest.mark.parametrize("path,expected", [
    ("NOTAS_INFORME.md", True), ("models/vision/resnet18.pt", True), ("data/raw/rocole/img.jpg", True),
    ("data.zip", True), ("src/vision/infer.py", False), ("models/yield_model.pkl", False), ("data/raw/varieties.csv", False),
])
def test_forbidden_paths(path, expected):
    assert (publish_check.forbidden_reason(path) is not None) is expected


def test_row_normalize_handles_empty_rows():
    out = figures.row_normalize(np.array([[2, 2], [0, 0]]))
    assert out.tolist() == [[0.5, 0.5], [0.0, 0.0]]


def test_loso_change_uses_mean_of_seeds_and_regions():
    rows = []
    for region in ("a", "b"):
        for predictor, mae in (("preentrenado|forecast_gru", 2.0), ("preentrenado|forecast_gru_s1", 2.0),
                               ("loso_ft|forecast_gru", 1.0), ("loso_ft|forecast_gru_s1", 1.0), ("clima_era5", 9.0)):
            rows.append({"region": region, "variable": "precipitation_sum", "horizonte": 1, "predictor": predictor, "mae": mae})
    out = figures.loso_change(pd.DataFrame(rows))
    r = out[(out["variable"] == "precipitation_sum") & (out["horizonte"] == 1)].iloc[0]
    assert r["cambio_pct"] == pytest.approx(-50.0)


def test_confusion_figure_is_written(tmp_path):
    exp = tmp_path / "experiments"
    (exp / "vision").mkdir(parents=True)
    m = [[5, 1, 0, 0], [0, 4, 1, 0], [0, 1, 3, 0], [0, 0, 0, 6]]
    (exp / "vision" / "confusion_hoja.json").write_text(json.dumps({"candidato": "x", "test_bracol": m, "test_rocole": m}), encoding="utf-8")
    figures._style()
    path = figures.fig_confusion(exp, tmp_path / "out")
    assert path.exists() and path.stat().st_size > 1000


def test_missing_results_fail_with_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="docs/reproducir.md"):
        figures.fig_voi(tmp_path, tmp_path / "out")

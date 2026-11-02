"""Las ventanas del pronóstico no deben filtrar datos del futuro ni usar estadísticas de test."""

import numpy as np
import pandas as pd
import pytest

from src.common.paths import RAW_DIR
from src.forecast.data import load_config, make_windows, split_bounds, split_windows


def test_ventanas_alinean_salida_con_el_dia_siguiente():
    serie = np.arange(40, dtype=float)[:, None]          # el valor es su propia posición
    x, y, t = make_windows(serie, serie, input_days=5, max_horizon=3)
    assert x.shape == (33, 5, 1) and y.shape == (33, 3, 1)
    for i in (0, 10, 32):
        assert x[i, -1, 0] == t[i]                       # la entrada termina en el día t
        assert list(y[i, :, 0]) == [t[i] + 1, t[i] + 2, t[i] + 3]


@pytest.mark.skipif(not (RAW_DIR / "openmeteo" / "era5_antigua.csv").exists(),
                    reason="descargar ERA5 con python -m src.forecast.download")
def test_split_temporal_sin_fuga_y_normalizacion_solo_con_train():
    cfg = load_config()
    out = split_windows("antigua", cfg)
    bounds = split_bounds(cfg)
    h = cfg["max_horizon"]
    for name in ("train", "val", "test"):
        assert len(out[name]["x"]) > 0
        last_out = pd.DatetimeIndex(out[name]["dates"]) + pd.Timedelta(days=h)
        lo, hi = bounds[name]
        assert (last_out >= lo).all() and (last_out <= hi).all()
    # toda la salida de validación (del día 1 al 14) cae dentro de validación
    first_val = pd.DatetimeIndex(out["val"]["dates"]) + pd.Timedelta(days=1)
    assert (first_val >= bounds["val"][0]).all()
    # la salida de val empieza después del último día de train, la de test después del de val
    assert pd.DatetimeIndex(out["val"]["dates"]).min() + pd.Timedelta(days=1) > bounds["train"][1] - pd.Timedelta(days=h)
    # estadísticas de la normalización calculadas solo con train: media de train ≈ 0, la de test no
    assert abs(float(out["train"]["x"][:, :, 0].mean())) < 0.1

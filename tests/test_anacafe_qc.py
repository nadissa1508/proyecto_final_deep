"""El control de calidad de ANACAFE debe anular sensores muertos sin tocar datos buenos."""

import numpy as np
import pandas as pd

from src.forecast.anacafe import quality_control


def _station(days=120, **overrides):
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "date": pd.date_range("2026-05-01", periods=days), "station_id": 1, "name": "x",
        "lon": -90.0, "lat": 15.0,
        "rain_mm": rng.gamma(0.8, 6, days), "rh_mean": rng.uniform(70, 95, days),
        "wind_kph": rng.uniform(0.5, 3, days), "t_mean": rng.uniform(15, 22, days),
        "dew_mean": rng.uniform(10, 17, days),
    })
    df["t_max"], df["t_min"] = df["t_mean"] + 4, df["t_mean"] - 4
    for col, values in overrides.items():
        df[col] = values
    return df


def test_viento_en_cero_se_anula():
    wind = np.r_[np.full(30, 0.0), np.random.RandomState(1).uniform(0.5, 3, 90)]
    clean, flags = quality_control(_station(wind_kph=wind))
    assert flags["viento_en_cero"] == 0 or clean["wind_kph"].iloc[:30].isna().all()
    assert clean["wind_kph"].iloc[30:].notna().all()


def test_lluvia_cero_en_temporada_de_lluvias_se_anula():
    clean, flags = quality_control(_station(rain_mm=np.zeros(120)))
    assert clean["rain_mm"].isna().all()
    assert flags["lluvia_cero_en_temporada_de_lluvias"] == 120


def test_sequia_corta_no_se_toca():
    rain = np.random.RandomState(2).gamma(0.8, 6, 120)
    rain[40:60] = 0
    clean, _ = quality_control(_station(rain_mm=rain))
    assert clean["rain_mm"].notna().all()

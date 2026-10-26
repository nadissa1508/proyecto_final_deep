"""Pruebas de `src/forecast/infer.py` con modelos sintéticos pequeños (no dependen de los pesos reales)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.forecast.data import INPUT_FEATURES
from src.forecast.infer import ModelNotAvailable, WeatherForecaster
from src.forecast.models import build_model

TARGETS = ["temperature_2m_mean", "relative_humidity_2m_mean", "precipitation_sum"]
CONFIG = {"model": "gru", "hidden": 8, "layers": 1, "dropout": 0.0}
COLUMNS = INPUT_FEATURES + ["doy_sin", "doy_cos"]


def make_artifact(tmp_path, min_horizon: int = 7):
    torch.manual_seed(0)
    members = {}
    for kind in ("base", "ft"):
        entries = []
        for i in range(2):
            path = tmp_path / f"{kind}{i}.pt"
            torch.save(build_model(CONFIG, len(COLUMNS), 3, 14).state_dict(), path)
            entries.append({"name": f"{kind}{i}", "weights": str(path), "config": CONFIG})
        members[kind] = entries
    norm = {"mean": {c: 0.0 for c in COLUMNS}, "std": {c: 1.0 for c in COLUMNS}}
    spec = {"features": INPUT_FEATURES, "input_days": 90, "max_horizon": 14, "targets": TARGETS,
            "sources": {TARGETS[0]: "base", TARGETS[1]: "ft", TARGETS[2]: "ft"}, "members": members,
            "normalizers": {"r": norm}, "stale_fill_days": 5, "min_horizon_with_gap": min_horizon}
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def history(days: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=days, freq="D")
    return pd.DataFrame(np.abs(rng.normal(5, 1, (days, len(INPUT_FEATURES)))), index=idx, columns=INPUT_FEATURES)


def test_forecast_shapes_and_physical_ranges(tmp_path):
    result = WeatherForecaster(make_artifact(tmp_path)).predict(history(), "r")
    assert len(result.days) == 14 and result.stale_days == 0 and not any(result.degraded)
    for target in TARGETS:
        assert len(result.values[target]) == 14 and len(result.spread[target]) == 14
    assert min(result.values["precipitation_sum"]) >= 0


def test_stale_history_flags_short_horizons(tmp_path):
    f = WeatherForecaster(make_artifact(tmp_path))
    h = history()
    result = f.predict(h, "r", as_of=(h.index[-1] + pd.Timedelta(days=5)).date())
    assert result.stale_days == 5
    assert result.degraded == [True] * 6 + [False] * 8
    assert result.days[0] == (h.index[-1] + pd.Timedelta(days=6)).date().isoformat()
    assert result.notes


def test_rejects_short_nulls_and_unknown_region(tmp_path):
    f = WeatherForecaster(make_artifact(tmp_path))
    with pytest.raises(ValueError):
        f.predict(history(50), "r")
    bad = history()
    bad.iloc[10, 0] = np.nan
    with pytest.raises(ValueError):
        f.predict(bad, "r")
    with pytest.raises(ValueError):
        f.predict(history(), "otra")


def test_missing_artifact_or_weights(tmp_path):
    with pytest.raises(ModelNotAvailable):
        WeatherForecaster(tmp_path / "no_existe.json")
    path = make_artifact(tmp_path)
    (tmp_path / "ft1.pt").unlink()
    with pytest.raises(ModelNotAvailable):
        WeatherForecaster(path)

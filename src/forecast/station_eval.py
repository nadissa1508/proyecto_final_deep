"""Evalúa el pronóstico contra las estaciones de ANACAFE (clima medido, no ERA5).

Verdad: la estación (con el control de calidad de `anacafe.py`), en el día objetivo t+h. Predictores,
todos emitidos con la información disponible hasta el día t:

- `persistencia_era5`: el valor de ERA5 de hoy.
- `persistencia_estacion`: el valor medido en la estación hoy (requiere tener estación).
- `clima_era5`: climatología mediana de ERA5 (1980–2017) del día del año objetivo.
- `numerico`: pronóstico numérico de Open-Meteo emitido h días antes (solo h ≤ 7, desde 2024).
- `modelo`: LSTM / GRU / Transformer entrenados con ERA5 (ERA5 como entrada, sin afinar).
- `<predictor>+sesgo`: el mismo predictor corregido con el sesgo medio contra la estación, calculado SOLO en
  el periodo de ajuste (`stations.adjust_*` de `configs/forecast.yaml`). Es el control justo para el afinado:
  el afinado debe superar a "modelo preentrenado + corrección de sesgo", no solo al modelo crudo.

Todas las filas comparadas en cada (región, variable, horizonte) son las mismas: se usa solo el día objetivo
en que **todos** los predictores y la estación tienen dato. El test es siempre posterior al periodo de ajuste
(día objetivo ≥ `stations.test_start`).

Salida: `experiments/forecast/station_eval.csv` (una fila por región, variable, horizonte y predictor).

Uso:
    python -m src.forecast.station_eval
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

from src.common.paths import EXPERIMENTS_DIR, MODELS_DIR
from src.forecast.anacafe import load_clean
from src.forecast.baselines import LEAD_COLUMN, climatology
from src.forecast.data import all_windows, load_config, load_region, split_bounds
from src.forecast.download import load_regions
from src.forecast.models import build_model
from src.forecast.train import predict, to_physical
from src.common.paths import RAW_DIR

OUT = EXPERIMENTS_DIR / "forecast" / "station_eval.csv"
STATION_COL = {"temperature_2m_mean": "t_mean", "relative_humidity_2m_mean": "rh_mean",
               "precipitation_sum": "rain_mm"}


def load_run_configs() -> List[dict]:
    """Configuración de cada corrida del pronóstico base (se excluye el barrido, que vive en otra carpeta)."""
    configs = []
    for run in sorted((EXPERIMENTS_DIR / "forecast").iterdir()):
        if (run / "config.yaml").exists() and (run / "results.json").exists() and "smoke" not in run.name:
            configs.append(yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")))
    return configs


def load_models(cfgs: List[dict], n_features: int, n_targets: int, horizon: int) -> Dict[str, torch.nn.Module]:
    models = {}
    for cfg in cfgs:
        model = build_model(cfg, n_features, n_targets, horizon)
        model.load_state_dict(torch.load(MODELS_DIR / "forecast" / f"{cfg['name']}.pt", map_location="cpu",
                                         weights_only=True))
        models[cfg["name"]] = model.eval()
    return models


def _bias(truth: np.ndarray, pred: np.ndarray, adjust: np.ndarray) -> float:
    ok = adjust & ~np.isnan(truth) & ~np.isnan(pred)
    return float((truth[ok] - pred[ok]).mean()) if ok.any() else float("nan")


def evaluate_region(region: str, spec: dict, fc: dict, models: Dict[str, torch.nn.Module],
                    stations: pd.DataFrame) -> pd.DataFrame:
    st = fc["stations"]
    adjust_lo, adjust_hi, test_lo = pd.Timestamp(st["adjust_start"]), pd.Timestamp(st["adjust_end"]), pd.Timestamp(st["test_start"])
    df = load_region(region)
    w = all_windows(region, fc)
    t_idx, dates = w["t_index"], w["dates"]
    targets = fc["targets"]
    station = stations[stations["station_id"] == spec["estacion_anacafe"]].set_index("date").reindex(df.index)
    prev = pd.read_csv(RAW_DIR / "openmeteo" / f"previous_runs_{region}.csv", parse_dates=["date"]).set_index("date").reindex(df.index)
    preds = {name: to_physical(predict(m, w["x"], torch.device("cpu"), False), w["normalizer"], targets)
             for name, m in models.items()}
    bounds = split_bounds(fc)
    rows = []
    for j, target in enumerate(targets):
        era5 = df[target].to_numpy()
        truth_series = station[STATION_COL[target]].to_numpy()
        clim = climatology(df, target, bounds["train"][1], fc["seasonal_window_days"], stat="median")
        for h in fc["horizons"]:
            target_pos = t_idx + h
            target_dates = df.index[target_pos]
            truth = truth_series[target_pos]
            cols = {"persistencia_era5": era5[t_idx], "persistencia_estacion": truth_series[t_idx],
                    "clima_era5": clim.reindex(np.minimum(target_dates.dayofyear.to_numpy(), 365)).to_numpy()}
            lead = LEAD_COLUMN[target].format(h=h)
            if h <= 7:
                cols["numerico"] = prev[lead].to_numpy()[target_pos]
            for name, p in preds.items():
                cols[name] = p[:, h - 1, j]
            adjust = (target_dates >= adjust_lo) & (target_dates <= adjust_hi)
            test = target_dates >= test_lo
            for key in [k for k in cols if k.startswith(("clima", "numerico", "forecast_"))]:
                bias = _bias(truth, cols[key], adjust)
                corrected = cols[key] + bias
                cols[f"{key}+sesgo"] = np.clip(corrected, 0, None) if target == "precipitation_sum" else corrected
            valid = test & ~np.isnan(truth)
            for v in cols.values():
                valid &= ~np.isnan(v)
            if valid.sum() < 30:
                continue
            for key, v in cols.items():
                rows.append({"region": region, "variable": target, "horizonte": h, "n": int(valid.sum()),
                             "predictor": key, "mae": float(np.abs(v[valid] - truth[valid]).mean())})
    return pd.DataFrame(rows)


def main() -> None:
    fc = load_config()
    stations, _ = load_clean()
    cfgs = load_run_configs()
    sample = all_windows(next(iter(load_regions())), fc)
    models = load_models(cfgs, sample["x"].shape[2], len(fc["targets"]), fc["max_horizon"])
    table = pd.concat([evaluate_region(r, spec, fc, models, stations) for r, spec in load_regions().items()],
                      ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(OUT, index=False)
    table["familia"] = table["predictor"].str.replace(r"_s\d(?=\+|$)", "", regex=True)   # semillas -> una familia
    summary = table.groupby(["variable", "horizonte", "familia"])["mae"].mean().unstack("familia")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"Muestras comparadas por región (mín–máx): {table.groupby(['region','variable','horizonte'])['n'].first().min()}"
          f"–{table.groupby(['region','variable','horizonte'])['n'].first().max()}")
    for variable in fc["targets"]:
        print(f"\n== {variable}: MAE contra la estación, media de las regiones (test desde {fc['stations']['test_start']})")
        print(summary.loc[variable].round(2).T.to_string())


if __name__ == "__main__":
    main()

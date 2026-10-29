"""Construye la tabla de terciles y errores de ventana que usa `evidence.py` (`models/fusion/evidence.json`).

1. **Terciles**: se corre el pronosticador real (`WeatherForecaster`) sobre el historial de ERA5 de cada región, con un origen
   cada `--stride` días entre 1980 y 2017, y se guardan los terciles de temperatura media, humedad media y lluvia acumulada
   de la ventana de 7 y de 14 días, por región y mes del primer día pronosticado.
2. **Error de la ventana**: con orígenes de 2018–2020 (el periodo de validación del pronóstico) se compara la ventana
   pronosticada con la ventana observada en ERA5 y se guarda la desviación típica del error (sin el sesgo medio, que depende
   de que el modelo afinado predice la estación y no ERA5). Es el error que `evidence.py` reparte como incertidumbre; el de la
   lluvia se mide en `log1p(mm)`, la escala en que `evidence.py` la evalúa.

Uso:
    python -m src.fusion.build_evidence
    python -m src.fusion.build_evidence --stride 14      # prueba rápida
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from src.common.paths import MODELS_DIR
from src.forecast.data import load_config as load_forecast_config
from src.forecast.data import load_region
from src.forecast.download import load_regions
from src.forecast.infer import WeatherForecaster
from src.fusion.config import load_config
from src.fusion.evidence import HUMIDITY, RAIN, TEMPERATURE, VARS, window_from_values

HISTORY_DAYS = 120


def run_origins(forecaster: WeatherForecaster, region: str, df: pd.DataFrame, dates: pd.DatetimeIndex, horizons):
    """Pronostica desde cada origen y devuelve filas (fecha, mes, horizonte, T, H, P, T_obs, H_obs, P_obs)."""
    rows = []
    positions = df.index.get_indexer(dates)
    for t in positions:
        history = df.iloc[t - HISTORY_DAYS + 1: t + 1]
        values = forecaster.predict(history, region).values
        month = int(df.index[t + 1].month)
        for h in horizons:
            w = window_from_values(values, h, region, month)
            nxt = df.iloc[t + 1: t + 1 + h]
            rows.append((df.index[t], month, h, w.mean_temperature, w.mean_humidity, w.total_rain,
                         nxt[TEMPERATURE].mean(), nxt[HUMIDITY].mean(), nxt[RAIN].sum()))
    return pd.DataFrame(rows, columns=["origin", "month", "h", "T", "H", "P", "T_obs", "H_obs", "P_obs"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stride", type=int, default=7, help="días entre orígenes")
    args = parser.parse_args()
    cfg, fc = load_config(), load_forecast_config()
    horizons = cfg["horizons"]
    forecaster = WeatherForecaster()
    train_end, val_end = pd.Timestamp(fc["split"]["train_end"]), pd.Timestamp(fc["split"]["val_end"])
    thresholds: dict = {}
    residuals = defaultdict(list)
    counts = {}
    for region in load_regions():
        df = load_region(region)
        first = df.index[HISTORY_DAYS]
        hist_dates = df.index[(df.index >= first) & (df.index <= train_end - pd.Timedelta(days=max(horizons)))][:: args.stride]
        val_dates = df.index[(df.index > train_end) & (df.index <= val_end - pd.Timedelta(days=max(horizons)))][:: args.stride]
        hist = run_origins(forecaster, region, df, hist_dates, horizons)
        val = run_origins(forecaster, region, df, val_dates, horizons)
        thresholds[region] = {}
        for h in horizons:
            sub = hist[hist["h"] == h]
            thresholds[region][str(h)] = {
                str(m): {k: [float(sub.loc[sub["month"] == m, k].quantile(1 / 3)),
                             float(sub.loc[sub["month"] == m, k].quantile(2 / 3))] for k in VARS}
                for m in range(1, 13)}
            sv = val[val["h"] == h]
            for k in VARS:
                f, o = (np.log1p(sv[k]), np.log1p(sv[f"{k}_obs"])) if k == "P" else (sv[k], sv[f"{k}_obs"])
                residuals[(h, k)].append((f - o).to_numpy())
        counts[region] = {"origenes_terciles": int(len(hist_dates)), "origenes_error": int(len(val_dates))}
        print(f"{region}: {len(hist_dates)} orígenes de terciles, {len(val_dates)} de error", flush=True)
    sigma, bias = {}, {}
    for h in horizons:
        sigma[str(h)] = {k: float(np.concatenate(residuals[(h, k)]).std()) for k in VARS}
        bias[str(h)] = {k: float(np.concatenate(residuals[(h, k)]).mean()) for k in VARS}
    out = {"thresholds": thresholds, "sigma": sigma,
           "meta": {"stride_dias": args.stride, "terciles": "pronósticos 1980-2017 por región y mes",
                    "error": "pronóstico vs ERA5, orígenes 2018-2020, desv. típica sin sesgo", "sesgo_medio": bias,
                    "conteos": counts}}
    path = MODELS_DIR / "fusion" / "evidence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("desviación típica del error de la ventana:", json.dumps(sigma))
    print("sesgo medio (informativo):", json.dumps(bias))
    print("escrito", path)


if __name__ == "__main__":
    main()

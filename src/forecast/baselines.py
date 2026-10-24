"""Baselines del pronóstico: persistencia, estacional ingenuo y pronóstico numérico de Open-Meteo.

Para cada región, variable y horizonte h se mide el error absoluto medio (MAE, en unidades
físicas: °C, % y mm) sobre el periodo de test, comparando el valor pronosticado para el día
t+h con el valor ERA5 de ese día. La fecha de emisión t es siempre h días antes.

- **Persistencia:** el valor de hoy (día t) se repite para el día t+h.
- **Estacional ingenuo:** la climatología del día del año, promediada en ±`seasonal_window_days`,
  calculada **solo con el periodo de train**.
- **Pronóstico numérico (techo):** lo que el modelo meteorológico de Open-Meteo ya había
  pronosticado h días antes (`*_leadH`, solo h = 1..7 y solo desde 2024). Es otro modelo
  físico, con su propio sesgo respecto a ERA5; por eso se compara solo en el periodo en que
  existe, junto a los otros dos baselines evaluados en esos mismos días.

Salida: `experiments/forecast/baselines.csv` (versionado, evidencia para el informe).

Uso:
    python -m src.forecast.baselines
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.paths import EXPERIMENTS_DIR, RAW_DIR
from src.forecast.data import load_config, load_region, split_bounds
from src.forecast.download import load_regions

OUT = EXPERIMENTS_DIR / "forecast" / "baselines.csv"
LEAD_COLUMN = {"temperature_2m_mean": "temperature_2m_mean_lead{h}",
               "relative_humidity_2m_mean": "relative_humidity_2m_mean_lead{h}",
               "precipitation_sum": "precipitation_sum_lead{h}"}


def climatology(df: pd.DataFrame, target: str, train_end: pd.Timestamp, window: int) -> pd.Series:
    """Media por día del año con el periodo de train, suavizada en ±window días (circular)."""
    train = df.loc[:train_end, target]
    doy = np.minimum(train.index.dayofyear.to_numpy(), 365)   # el 29-feb y el 31-dic comparten día
    by_doy = train.groupby(doy).mean().reindex(range(1, 366))
    padded = pd.concat([by_doy.iloc[-window:], by_doy, by_doy.iloc[:window]])
    smooth = padded.rolling(2 * window + 1, center=True).mean().iloc[window:-window]
    return pd.Series(smooth.to_numpy(), index=range(1, 366))


def evaluate_region(region: str, cfg: dict) -> pd.DataFrame:
    df = load_region(region)
    bounds = split_bounds(cfg)
    test_lo = bounds["test"][0]
    prev = pd.read_csv(RAW_DIR / "openmeteo" / f"previous_runs_{region}.csv",
                       parse_dates=["date"]).set_index("date")
    rows = []
    for target in cfg["targets"]:
        clim = climatology(df, target, bounds["train"][1], cfg["seasonal_window_days"])
        for h in cfg["horizons"]:
            truth = df[target].shift(-h)                      # valor del día t+h, indexado en t
            days = df.index[(df.index + pd.Timedelta(days=h) >= test_lo) & truth.notna()]
            y, persist = truth.loc[days], df[target].loc[days]
            target_days = days + pd.Timedelta(days=h)
            seasonal = pd.Series(clim.reindex(np.minimum(target_days.dayofyear, 365)).to_numpy(), index=days)
            row = {"region": region, "variable": target, "horizonte": h, "n_test": len(days),
                   "mae_persistencia": round(float((persist - y).abs().mean()), 3),
                   "mae_estacional": round(float((seasonal - y).abs().mean()), 3)}
            lead_col = LEAD_COLUMN[target].format(h=h)
            if h <= 7 and lead_col in prev.columns:
                both = pd.DataFrame({"num": prev[lead_col].reindex(target_days).to_numpy(),
                                     "y": y.to_numpy(), "per": persist.to_numpy(),
                                     "sea": seasonal.to_numpy()}).dropna()
                row["n_numerico"] = len(both)
                row["mae_numerico"] = round(float((both["num"] - both["y"]).abs().mean()), 3)
                row["mae_persistencia_mismo_periodo"] = round(float((both["per"] - both["y"]).abs().mean()), 3)
                row["mae_estacional_mismo_periodo"] = round(float((both["sea"] - both["y"]).abs().mean()), 3)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    table = pd.concat([evaluate_region(r, cfg) for r in load_regions()], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    for target in cfg["targets"]:
        part = table[table["variable"] == target]
        print(f"\n== {target}  (MAE, test desde {split_bounds(cfg)['test'][0].date()})")
        print(part.pivot(index="region", columns="horizonte",
                         values=["mae_persistencia", "mae_estacional"]).round(2).to_string())
        num = part.dropna(subset=["mae_numerico"])
        if len(num):
            print("-- mismo periodo (desde 2024), horizontes 1-7, promedio de las 6 regiones:")
            print(num.groupby("horizonte")[["mae_persistencia_mismo_periodo",
                  "mae_estacional_mismo_periodo", "mae_numerico"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()

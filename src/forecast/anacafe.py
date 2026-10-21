"""Lectura de las estaciones de ANACAFE (portal ClimaCafé) y control de calidad.

Los CSV diarios se exportan a mano del portal y se guardan en `data/raw/anacafe/`.
Cada archivo es una estación y un año; el nombre es libre (`estacion_015_DIARIO_2026.csv`,
`ESTACION_071_DIARIO_2026.csv`). Se leen todos los que tengan `DIARIO` en el nombre y
se ignoran los `MENSUAL`.

Columnas del CSV -> columnas normalizadas:

    Dia -> date · Lluvia_mm -> rain_mm · HumedadMedia -> rh_mean · VtokphMedia -> wind_kph
    TempMedia -> t_mean · TempMax -> t_max · TempMin -> t_min · TempRocioMedia -> dew_mean
    RadSolarMax -> rad_max · HorasSol -> sun_hours

El control de calidad **no borra filas**: pone `NaN` en el valor dudoso y cuenta cuántos
cambió, para poder reportarlo. Reglas (cada una cubre un fallo visto en datos reales):

- `wind_kph`: rachas de ≥ 7 días exactamente en 0 = anemómetro sin lectura.
- `rain_mm`: racha de ≥ 45 días exactamente en 0 con la mitad o más de sus días entre
  mayo y octubre (temporada de lluvias) = pluviómetro sin lectura. Las sequías normales
  (canícula) duran semanas, no 45 días sin una gota.
- `t_mean`, `rh_mean`, `dew_mean`: rachas de ≥ 5 días con exactamente el mismo valor =
  sensor congelado.
- Rangos físicos: humedad fuera de 0–100, lluvia negativa o > 500 mm/día, temperatura
  fuera de -5–45 °C, y `t_min <= t_mean <= t_max` (con 0.5 °C de tolerancia).

Uso:
    python -m src.forecast.anacafe       # imprime el reporte de calidad por estación
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.common.paths import RAW_DIR

ANACAFE_DIR = RAW_DIR / "anacafe"

COLUMNS = {
    "Dia": "date", "ID_ESTACION": "station_id", "NOMBRE": "name", "LONGITUD": "lon",
    "LATITUD": "lat", "Lluvia_mm": "rain_mm", "HumedadMedia": "rh_mean",
    "VtokphMedia": "wind_kph", "TempMedia": "t_mean", "TempMax": "t_max",
    "TempMin": "t_min", "TempRocioMedia": "dew_mean", "RadSolarMax": "rad_max",
    "HorasSol": "sun_hours",
}
VARIABLES = ["rain_mm", "rh_mean", "wind_kph", "t_mean", "t_max", "t_min",
             "dew_mean", "rad_max", "sun_hours"]

WIND_ZERO_RUN = 7
RAIN_ZERO_RUN = 45
STUCK_RUN = 5
WET_MONTHS = (5, 6, 7, 8, 9, 10)


def _runs(mask: pd.Series) -> list:
    """Intervalos (inicio, fin) de posiciones consecutivas donde `mask` es True."""
    out, start = [], None
    values = mask.to_numpy()
    for i, v in enumerate(values):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(values) - 1))
    return out


def load_raw(directory: Path = ANACAFE_DIR) -> pd.DataFrame:
    files = sorted(p for p in directory.glob("*.csv") if "DIARIO" in p.name.upper())
    if not files:
        raise FileNotFoundError(f"No hay CSV diarios en {directory}")
    frames = [pd.read_csv(p, parse_dates=["Dia"]).rename(columns=COLUMNS) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["station_id", "date"]).sort_values(["station_id", "date"])
    return df.reset_index(drop=True)


def quality_control(station: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Devuelve la serie de una estación con los valores dudosos en NaN y el conteo por regla."""
    s = station.set_index("date").asfreq("D").copy()
    s[["station_id", "name", "lon", "lat"]] = station[["station_id", "name", "lon", "lat"]].iloc[0].values
    flags: Dict[str, int] = {}

    def mark(rule: str, column: str, positions: np.ndarray) -> None:
        idx = s.index[positions]
        newly = s.loc[idx, column].notna().sum()
        s.loc[idx, column] = np.nan
        flags[rule] = flags.get(rule, 0) + int(newly)

    wind_zero = s["wind_kph"] == 0
    for a, b in _runs(wind_zero):
        if b - a + 1 >= WIND_ZERO_RUN:
            mark("viento_en_cero", "wind_kph", np.arange(a, b + 1))

    rain_zero = s["rain_mm"] == 0
    for a, b in _runs(rain_zero):
        days = s.index[a:b + 1]
        if len(days) >= RAIN_ZERO_RUN and np.isin(days.month, WET_MONTHS).mean() >= 0.5:
            mark("lluvia_cero_en_temporada_de_lluvias", "rain_mm", np.arange(a, b + 1))

    for col in ("t_mean", "rh_mean", "dew_mean"):
        same = s[col].diff().eq(0)
        for a, b in _runs(same):
            if b - a + 2 >= STUCK_RUN:
                mark("valor_congelado", col, np.arange(a - 1, b + 1))

    mark("humedad_fuera_de_rango", "rh_mean", np.flatnonzero(~s["rh_mean"].between(0, 100) & s["rh_mean"].notna()))
    mark("lluvia_imposible", "rain_mm", np.flatnonzero(((s["rain_mm"] < 0) | (s["rain_mm"] > 500)).to_numpy()))
    for col in ("t_mean", "t_max", "t_min", "dew_mean"):
        mark("temperatura_fuera_de_rango", col, np.flatnonzero(~s[col].between(-5, 45) & s[col].notna()))
    incoherent = (s["t_mean"] < s["t_min"] - 0.5) | (s["t_mean"] > s["t_max"] + 0.5)
    for col in ("t_mean", "t_max", "t_min"):
        mark("temperatura_incoherente", col, np.flatnonzero(incoherent.fillna(False).to_numpy()))
    return s.reset_index(), flags


def load_clean(directory: Path = ANACAFE_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Todas las estaciones con QC aplicado y una tabla-reporte por estación."""
    raw = load_raw(directory)
    clean, report = [], []
    for sid, g in raw.groupby("station_id"):
        c, flags = quality_control(g)
        clean.append(c)
        missing_days = int(c["t_mean"].isna().sum())
        report.append({
            "estacion": sid, "nombre": g["name"].iloc[0], "inicio": g["date"].min().date(),
            "fin": g["date"].max().date(), "dias_con_fila": len(g),
            "dias_calendario": len(c), "huecos": len(c) - len(g),
            **{f"qc_{k}": v for k, v in flags.items()},
            "t_mean_validos_pct": round(100 * (1 - missing_days / len(c)), 1),
            "lluvia_validos_pct": round(100 * c["rain_mm"].notna().mean(), 1),
            "viento_validos_pct": round(100 * c["wind_kph"].notna().mean(), 1),
        })
    return pd.concat(clean, ignore_index=True), pd.DataFrame(report).fillna(0)


def main() -> None:
    _, report = load_clean()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()

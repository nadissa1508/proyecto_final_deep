"""Compara ERA5 (Open-Meteo) y NASA POWER contra las estaciones de ANACAFE.

Para cada región cruza, día por día, la estación (con control de calidad) con las dos
fuentes en grilla y mide, solo en los días en que la estación tiene dato válido:

- temperatura media y humedad relativa: sesgo medio (fuente − estación) y correlación diaria;
- lluvia: cociente de totales (fuente / estación), correlación diaria y correlación de
  totales mensuales (la lluvia diaria de una grilla de ~25–50 km nunca empata día a día).

Periodo común: 2022-01-01 a 2025-12-31, el más completo de las 6 estaciones. ERA5 ya viene
con la altitud de la estación; POWER no admite altitud, así que su temperatura se compara tal cual.

Salida: `experiments/forecast/source_comparison.csv` (versionado, es evidencia del informe).

Uso:
    python -m src.forecast.compare_sources
"""

from __future__ import annotations

import pandas as pd

from src.common.paths import EXPERIMENTS_DIR, RAW_DIR
from src.forecast.anacafe import load_clean
from src.forecast.download import load_regions

START, END = "2022-01-01", "2025-12-31"
OUT = EXPERIMENTS_DIR / "forecast" / "source_comparison.csv"

# columna de la estación -> (columna ERA5, columna POWER)
VARIABLES = {
    "t_mean": ("temperature_2m_mean", "T2M"),
    "rh_mean": ("relative_humidity_2m_mean", "RH2M"),
    "rain_mm": ("precipitation_sum", "PRECTOTCORR"),
}


def _read(path, index_col: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=[index_col]).set_index(index_col)


def compare() -> pd.DataFrame:
    stations, _ = load_clean()
    rows = []
    for key, region in load_regions().items():
        s = stations[stations["station_id"] == region["estacion_anacafe"]].set_index("date").loc[START:END]
        era5 = _read(RAW_DIR / "openmeteo" / f"era5_{key}.csv", "date").loc[START:END]
        power = _read(RAW_DIR / "power" / f"power_{key}.csv", "date").loc[START:END]
        for var, (era_col, pow_col) in VARIABLES.items():
            for source, frame, col in (("ERA5", era5, era_col), ("POWER", power, pow_col)):
                j = pd.concat([s[var].rename("est"), frame[col].rename("src")], axis=1).dropna()
                row = {"region": key, "estacion": region["estacion_anacafe"], "variable": var,
                       "fuente": source, "dias": len(j), "corr_diaria": round(j["est"].corr(j["src"]), 3)}
                if var == "rain_mm":
                    row["cociente_total"] = round(j["src"].sum() / j["est"].sum(), 2)
                    monthly = j.resample("MS").sum()
                    monthly = monthly[j["est"].resample("MS").count() >= 25]
                    row["corr_mensual"] = round(monthly["est"].corr(monthly["src"]), 3)
                else:
                    row["sesgo"] = round((j["src"] - j["est"]).mean(), 2)
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    table = compare()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    for var in VARIABLES:
        part = table[table["variable"] == var].dropna(axis=1, how="all").drop(columns="variable")
        print(f"\n== {var}  (periodo {START} a {END})")
        print(part.pivot(index="region", columns="fuente", values=[c for c in part.columns
              if c not in ("region", "estacion", "fuente", "dias")]).to_string())


if __name__ == "__main__":
    main()

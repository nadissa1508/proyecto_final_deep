"""Descarga de NASA POWER (diario, desde 1981) para las regiones de `configs/regions.yaml`.

Segunda fuente climática, independiente de Open-Meteo, para contrastar:
POWER usa MERRA-2 y satélite (GPM IMERG para lluvia), con grilla de ~0.5° x 0.625°,
y no tiene cupos que importen. La consulta es por coordenadas, así que sirve
para Guatemala aunque la documentación no liste países.

Variables: T2M, T2M_MAX, T2M_MIN, T2MDEW, RH2M, PRECTOTCORR (lluvia mm/día),
ALLSKY_SFC_SW_DWN (radiación) y WS2M (viento a 2 m). El valor -999 es "sin dato".

POWER no recibe altitud: su temperatura corresponde a la altura media de su
celda. Para compararla con una finca hay que corregirla por altitud o
contrastarla solo en variaciones (anomalías), no en nivel absoluto.

Salida (no versionada): `data/raw/power/power_<region>.csv`.

Uso:
    python -m src.forecast.download_power
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from src.common.paths import RAW_DIR
from src.forecast.download import check_regions, load_regions

OUT_DIR = RAW_DIR / "power"
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = ("T2M", "T2M_MAX", "T2M_MIN", "T2MDEW", "RH2M", "PRECTOTCORR",
              "ALLSKY_SFC_SW_DWN", "WS2M")
START_YEAR = 1981
FILL = -999.0


def fetch(lat: float, lon: float, start: date, end: date, retries: int = 5) -> pd.DataFrame:
    params = {"parameters": ",".join(PARAMETERS), "community": "AG", "latitude": lat,
              "longitude": lon, "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
              "format": "JSON"}
    for attempt in range(retries):
        try:
            resp = requests.get(URL, params=params, timeout=180)
        except requests.exceptions.RequestException:
            time.sleep(15 * (attempt + 1))
            continue
        if resp.status_code == 200:
            df = pd.DataFrame(resp.json()["properties"]["parameter"])
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
            return df.astype(float).replace(FILL, np.nan)
        if resp.status_code in (429, 500, 502, 503):
            time.sleep(30 * (attempt + 1))
            continue
        raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
    raise RuntimeError("NASA POWER no respondió")


def main() -> None:
    regions = load_regions()
    check_regions(regions)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    end = date.today() - timedelta(days=5)
    for key, region in regions.items():
        frames = []
        for year in range(START_YEAR, end.year + 1, 10):
            block_end = min(date(year + 9, 12, 31), end)
            frames.append(fetch(region["lat"], region["lon"], date(year, 1, 1), block_end))
        df = pd.concat(frames).sort_index()
        # La radiación solar de POWER empieza en 1984 (satélite CERES): 1981-1983 faltan por diseño.
        checked = df.copy()
        checked.loc[checked.index < "1984-01-01", "ALLSKY_SFC_SW_DWN"] = 0.0
        nulls = checked.isna().mean()
        if (nulls > 0.01).any():
            raise RuntimeError(f"{key}: variables con >1 % de nulos: {nulls[nulls > 0.01].round(3).to_dict()}")
        df.rename_axis("date").to_csv(OUT_DIR / f"power_{key}.csv")
        print(f"[power] {key}: {len(df)} días ({df.index.min().date()} → {df.index.max().date()}), "
              f"lluvia anual media {df['PRECTOTCORR'].sum() / (len(df) / 365.25):.0f} mm", flush=True)


if __name__ == "__main__":
    main()

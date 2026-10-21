"""Descarga de datos climáticos de Open-Meteo para las regiones de `configs/regions.yaml`.

Dos fuentes:

1. **Histórico (ERA5 + ERA5-Land, diario, desde 1980)** desde
   `archive-api.open-meteo.com` con `models=era5_seamless`. Se probó
   `era5_land` solo y **devuelve `null` en todos los días de lluvia y
   evapotranspiración** en las coordenadas de Guatemala (verificado con
   Huehuetenango). `era5_seamless` usa ERA5-Land para temperatura y humedad
   y ERA5 (25 km) para lluvia y evapotranspiración. Es la serie de
   entrenamiento de los modelos de pronóstico. Se empieza en 1980
   porque antes de la era satelital (1979) el reanálisis tiene muchas menos
   observaciones asimiladas, y porque así la descarga completa cabe en el cupo
   diario gratuito. ERA5-Land llega con ~5 días de retraso: la fecha final es
   hoy - 7 días.
2. **Pronósticos ya emitidos (Previous Runs API)** desde
   `previous-runs-api.open-meteo.com`. Para cada día válido trae lo que el
   modelo numérico pronosticó 1..7 días antes. Solo existe en horario; aquí se
   agrega a diario (media de temperatura y humedad, suma de lluvia) en hora
   local. Es la referencia "techo" de nuestros modelos a 1–7 días.

Cupo gratuito de Open-Meteo (uso no comercial): 600 llamadas/min, 5,000/hora,
10,000/día. Las llamadas se ponderan: 1 llamada = 1 ubicación, ≤ 10 variables y
≤ 14 días; más días o variables cuentan proporcionalmente. Este script estima
el peso de cada petición y espera lo necesario para no pasar de ~4,000/hora.
Cada bloque descargado se guarda en `data/raw/openmeteo/_cache/`, así que si
se corta (internet, cupo diario) basta volver a correrlo y continúa.

Salida en `data/raw/openmeteo/` (no versionado):
- `era5_<region>.csv`, `previous_runs_<region>.csv`, `metadata.json`.

Uso:
    python -m src.forecast.download            # todo
    python -m src.forecast.download --only era5
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import yaml

from src.common.paths import CONFIGS_DIR, RAW_DIR

OUT_DIR = RAW_DIR / "openmeteo"
CACHE_DIR = OUT_DIR / "_cache"
TIMEZONE = "America/Guatemala"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

ERA5_START = date(1980, 1, 1)
ERA5_MODEL = "era5_seamless"
DAILY_VARS = (
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "dew_point_2m_mean",
    "precipitation_sum",
    "precipitation_hours",
    "et0_fao_evapotranspiration",
)

PREV_START = date(2024, 1, 1)
PREV_BASE_VARS = ("temperature_2m", "relative_humidity_2m", "precipitation")
LEADS = range(1, 8)

HOURLY_BUDGET = 4000.0


class DailyQuotaExceeded(RuntimeError):
    """Open-Meteo rechazó por cupo diario: hay que reintentar otro día."""


def load_regions() -> Dict[str, dict]:
    with open(CONFIGS_DIR / "regions.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["regions"]


REQUIRED_FIELDS = ("lat", "lon", "elevation_m")


def check_regions(regions: Dict[str, dict]) -> None:
    """Exige coordenadas y altitud reales: sin ellas Open-Meteo usa el relieve del punto,
    y la temperatura sale como la de ese punto aunque la finca esté 900 m más arriba."""
    missing = {k: [f for f in REQUIRED_FIELDS if r.get(f) is None] for k, r in regions.items()}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise SystemExit("Faltan datos en configs/regions.yaml (copiarlos del portal de ANACAFE, "
                         f"icono de mapa de la estación): {missing}")


def _loc(region: dict) -> str:
    return f"{region['lat']}_{region['lon']}_{region['elevation_m']}"


def _loc_params(region: dict) -> dict:
    """`elevation` hace que Open-Meteo corrija la temperatura a la altitud de la finca."""
    return {"latitude": region["lat"], "longitude": region["lon"], "elevation": region["elevation_m"]}


def call_weight(n_days: int, n_vars: int) -> float:
    return (n_days / 14.0) * max(1.0, n_vars / 10.0)


def _get(url: str, params: dict, weight: float, retries: int = 8) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=180)
        except requests.exceptions.RequestException as exc:
            print(f"   conexión fallida ({exc.__class__.__name__}), reintento {attempt + 1}", flush=True)
            time.sleep(30 * (attempt + 1))
            continue
        if resp.status_code == 200:
            try:
                payload = resp.json()
            except ValueError:
                print(f"   respuesta 200 sin JSON ({len(resp.content)} bytes), reintento {attempt + 1}", flush=True)
                time.sleep(30 * (attempt + 1))
                continue
            time.sleep(weight / HOURLY_BUDGET * 3600)
            return payload
        reason = resp.json().get("reason", resp.text[:200]) if resp.headers.get(
            "content-type", "").startswith("application/json") else resp.text[:200]
        if resp.status_code == 429:
            if "Daily" in reason:
                raise DailyQuotaExceeded(reason)
            wait = 3600 if "Hourly" in reason else 65
            print(f"   límite de la API ({reason}); espero {wait} s", flush=True)
            time.sleep(wait)
            continue
        if resp.status_code >= 500:
            time.sleep(30 * (attempt + 1))
            continue
        raise RuntimeError(f"{resp.status_code} {resp.url}: {reason}")
    raise RuntimeError(f"Sin respuesta tras {retries} intentos: {url}")


def _blocks(start: date, end: date, days: int) -> List[tuple]:
    out = []
    while start <= end:
        block_end = min(start + timedelta(days=days - 1), end)
        out.append((start, block_end))
        start = block_end + timedelta(days=1)
    return out


def _cached_json(path: Path, url: str, params: dict, weight: float) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = _get(url, params, weight)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def download_era5(key: str, region: dict, end: date) -> tuple:
    frames, grid = [], {}
    for b_start, b_end in _blocks(ERA5_START, end, days=5 * 365):
        cache = CACHE_DIR / f"{ERA5_MODEL}_{key}_{_loc(region)}_{b_start}_{b_end}.json"
        weight = call_weight((b_end - b_start).days + 1, len(DAILY_VARS))
        data = _cached_json(cache, ARCHIVE_URL, {
            **_loc_params(region),
            "start_date": b_start.isoformat(), "end_date": b_end.isoformat(),
            "daily": ",".join(DAILY_VARS), "models": ERA5_MODEL,
            "timezone": TIMEZONE,
        }, weight)
        if abs(data["elevation"] - region["elevation_m"]) > 1:
            raise RuntimeError(f"{key}: Open-Meteo usó {data['elevation']} m, se pidieron {region['elevation_m']} m")
        frames.append(pd.DataFrame(data["daily"]))
        grid = {k: data[k] for k in ("latitude", "longitude", "elevation")}
        print(f"   [era5] {key} {b_start}..{b_end} ok", flush=True)
    df = pd.concat(frames, ignore_index=True).rename(columns={"time": "date"})
    validate_era5(key, df)
    return df, grid


def validate_era5(key: str, df: pd.DataFrame) -> None:
    """Falla si falta alguna variable o hay huecos: un CSV vacío no debe pasar en silencio."""
    expected = (df["date"].astype("datetime64[ns]").max() - df["date"].astype("datetime64[ns]").min()).days + 1
    if len(df) != expected:
        raise RuntimeError(f"{key}: faltan fechas ({len(df)} de {expected})")
    nulls = df.isna().mean()
    bad = nulls[nulls > 0.01]
    if not bad.empty:
        raise RuntimeError(f"{key}: variables con >1 % de nulos: {bad.round(3).to_dict()}")
    if not 200 < df["precipitation_sum"].sum() / (len(df) / 365.25) < 6000:
        raise RuntimeError(f"{key}: lluvia anual fuera de rango plausible")


def download_previous_runs(key: str, region: dict, end: date) -> pd.DataFrame:
    hourly = [f"{v}_previous_day{d}" for v in PREV_BASE_VARS for d in LEADS]
    frames = []
    for b_start, b_end in _blocks(PREV_START, end, days=90):
        cache = CACHE_DIR / f"prev_{key}_{_loc(region)}_{b_start}_{b_end}.json"
        weight = call_weight((b_end - b_start).days + 1, len(hourly))
        data = _cached_json(cache, PREVIOUS_RUNS_URL, {
            **_loc_params(region),
            "start_date": b_start.isoformat(), "end_date": b_end.isoformat(),
            "hourly": ",".join(hourly), "timezone": TIMEZONE,
        }, weight)
        frames.append(pd.DataFrame(data["hourly"]))
        print(f"   [previous] {key} {b_start}..{b_end} ok", flush=True)
    h = pd.concat(frames, ignore_index=True)
    h["date"] = pd.to_datetime(h["time"]).dt.date
    agg = {}
    for d in LEADS:
        agg[f"temperature_2m_mean_lead{d}"] = (f"temperature_2m_previous_day{d}", "mean")
        agg[f"relative_humidity_2m_mean_lead{d}"] = (f"relative_humidity_2m_previous_day{d}", "mean")
        agg[f"precipitation_sum_lead{d}"] = (
            f"precipitation_previous_day{d}", lambda s: s.sum(min_count=24))
    return h.groupby("date").agg(**agg).reset_index()


def estimate_total_weight(n_regions: int, end: date) -> float:
    era5_days = (end - ERA5_START).days + 1
    prev_days = (end - PREV_START).days + 1
    per_region = call_weight(era5_days, len(DAILY_VARS)) + call_weight(
        prev_days, len(PREV_BASE_VARS) * len(LEADS))
    return n_regions * per_region


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", choices=["era5", "previous"], default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    regions = load_regions()
    check_regions(regions)
    end = date.today() - timedelta(days=7)
    meta_path = OUT_DIR / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    print(f"Peso estimado de la descarga completa: {estimate_total_weight(len(regions), end):.0f} "
          f"llamadas (cupo diario 10,000)", flush=True)

    try:
        for key, region in regions.items():
            if args.only in (None, "era5"):
                df, grid = download_era5(key, region, end)
                df.to_csv(OUT_DIR / f"era5_{key}.csv", index=False)
                metadata[key] = {**region, "grid": grid, "era5_start": ERA5_START.isoformat(),
                                 "era5_end": end.isoformat(), "era5_rows": len(df)}
                print(f"[era5] {key}: {len(df)} días, celda {grid}", flush=True)
            if args.only in (None, "previous"):
                prev = download_previous_runs(key, region, end)
                prev.to_csv(OUT_DIR / f"previous_runs_{key}.csv", index=False)
                metadata.setdefault(key, dict(region))["previous_rows"] = len(prev)
                print(f"[previous] {key}: {len(prev)} días", flush=True)
            metadata["_descarga"] = {"fecha": date.today().isoformat(),
                                     "fuente": "Open-Meteo (CC BY 4.0)"}
            meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    except DailyQuotaExceeded as exc:
        print(f"\nCUPO DIARIO AGOTADO: {exc}\nLo descargado quedó en caché. "
              "Volver a correr el mismo comando mañana para continuar.", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

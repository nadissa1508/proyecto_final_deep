"""Interfaz de inferencia del módulo de pronóstico: la única que usan la API y `src/fusion/`.

Recibe la **historia diaria** reciente de una finca (las 8 variables de ERA5 de Open-Meteo) y devuelve el
pronóstico de temperatura media, humedad relativa y lluvia para los siguientes 14 días. Lee un artefacto
(`models/forecast/artifact.json`) escrito por `export.py`: qué pesos usar por variable, los normalizadores y el
horizonte mínimo confiable. No contiene cifras de resultados.

Cómo se maneja el retraso de ERA5 (unos 5–8 días): si `as_of` es posterior al último día de la historia, los días
faltantes se rellenan repitiendo el último valor conocido y los horizontes menores a `min_horizon_with_gap` se
marcan como `degraded` (la validación mostró que ahí el error crece mucho; a 7–14 días casi no cambia).

Normalización: el modelo se entrenó con la media y desviación de ERA5 de cada región. `region` elige cuál usar;
para una finca nueva se usa la de la región más cercana (aproximación que se declara en la respuesta).

Si el artefacto o los pesos no existen se lanza `ModelNotAvailable` (la API responde 503).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import torch

from src.common.paths import MODELS_DIR
from src.forecast.data import INPUT_FEATURES, transform
from src.forecast.models import build_model

ARTIFACT_PATH = MODELS_DIR / "forecast" / "artifact.json"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ERA5_MODEL = "era5_seamless"
TIMEZONE = "America/Guatemala"


class ModelNotAvailable(RuntimeError):
    """No hay artefacto o faltan pesos: el módulo de pronóstico no está listo."""


@dataclass
class Forecast:
    region_used: str
    last_observed: str
    as_of: str
    stale_days: int
    days: List[str]
    values: Dict[str, List[float]]
    spread: Dict[str, List[float]]
    source: Dict[str, str]
    degraded: List[bool]
    notes: List[str] = field(default_factory=list)


def nearest_region(coords: Dict[str, dict], lat: float, lon: float) -> str:
    """Región (de las 6 con normalizador) más cercana; distancia plana, suficiente para escoger entre 6."""
    return min(coords, key=lambda r: (coords[r]["lat"] - lat) ** 2 + (coords[r]["lon"] - lon) ** 2)


def fetch_history(lat: float, lon: float, elevation_m: float, today: Optional[date] = None,
                  days: int = 120) -> pd.DataFrame:
    """Historia reciente de ERA5 (mismo producto y variables que el entrenamiento). Termina donde ERA5 ya publicó."""
    today = today or date.today()
    resp = requests.get(ARCHIVE_URL, params={
        "latitude": lat, "longitude": lon, "elevation": elevation_m, "models": ERA5_MODEL, "timezone": TIMEZONE,
        "start_date": (today - timedelta(days=days)).isoformat(), "end_date": today.isoformat(),
        "daily": ",".join(INPUT_FEATURES)}, timeout=60)
    resp.raise_for_status()
    frame = pd.DataFrame(resp.json()["daily"])
    frame["date"] = pd.to_datetime(frame.pop("time"))
    return frame.set_index("date").dropna(how="any")      # los últimos días aún no publicados vienen nulos


class WeatherForecaster:
    def __init__(self, artifact_path: Path = ARTIFACT_PATH, device: Optional[str] = None) -> None:
        if not artifact_path.exists():
            raise ModelNotAvailable(f"No existe el artefacto de pronóstico: {artifact_path}")
        self.spec = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.device = torch.device(device or "cpu")
        self.targets: List[str] = self.spec["targets"]
        self.input_days: int = self.spec["input_days"]
        self.horizon: int = self.spec["max_horizon"]
        n_features = len(INPUT_FEATURES) + 2
        self.models: Dict[str, List[torch.nn.Module]] = {}
        for kind in sorted(set(self.spec["sources"].values())):
            members = []
            for entry in self.spec["members"][kind]:
                path = MODELS_DIR / "forecast" / entry["weights"]
                if not path.exists():
                    raise ModelNotAvailable(f"Faltan los pesos {path}")
                model = build_model(entry["config"], n_features, len(self.targets), self.horizon)
                model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
                members.append(model.to(self.device).eval())
            self.models[kind] = members

    def _window(self, history: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        missing = [c for c in INPUT_FEATURES if c not in history]
        if missing:
            raise ValueError(f"A la historia le faltan columnas: {missing}")
        if history[INPUT_FEATURES].isna().any().any():
            raise ValueError("La historia tiene valores nulos")
        full = history[INPUT_FEATURES].reindex(pd.date_range(history.index[0], as_of, freq="D"))
        if full.iloc[: len(history)].isna().any().any():
            raise ValueError("La historia tiene fechas faltantes")
        full = full.ffill()                                   # días sin ERA5: se repite el último conocido
        if len(full) < self.input_days:
            raise ValueError(f"Se necesitan al menos {self.input_days} días de historia, hay {len(full)}")
        return full.iloc[-self.input_days:]

    @torch.no_grad()
    def predict(self, history: pd.DataFrame, region: str, as_of: Optional[date] = None) -> Forecast:
        if region not in self.spec["normalizers"]:
            raise ValueError(f"Región sin normalizador: {region}")
        last = history.index[-1]
        as_of_ts = pd.Timestamp(as_of) if as_of else last
        stale = int((as_of_ts - last).days)
        if stale < 0:
            raise ValueError("as_of es anterior al último día de la historia")
        window = self._window(history, as_of_ts)
        frame = transform(window)
        norm = self.spec["normalizers"][region]
        mean, std = pd.Series(norm["mean"])[frame.columns], pd.Series(norm["std"])[frame.columns]
        x = torch.from_numpy(((frame - mean) / std).to_numpy(dtype=np.float32)).unsqueeze(0).to(self.device)

        values, spread, source = {}, {}, {}
        for j, target in enumerate(self.targets):
            kind = self.spec["sources"][target]
            outs = np.stack([m(x)[0, :, j].cpu().numpy() for m in self.models[kind]])   # (miembros, horizonte)
            phys = outs * std[target] + mean[target]
            if target == "precipitation_sum":
                phys = np.clip(np.expm1(phys), 0, None)
            values[target] = [float(v) for v in phys.mean(axis=0)]
            spread[target] = [float(v) for v in phys.std(axis=0)]
            source[target] = kind
        days = [(as_of_ts + pd.Timedelta(days=h)).date().isoformat() for h in range(1, self.horizon + 1)]
        min_h = int(self.spec["min_horizon_with_gap"])
        degraded = [stale > 0 and h < min_h for h in range(1, self.horizon + 1)]
        notes = []
        if stale > 0:
            notes.append(f"La historia termina {stale} días antes de la fecha de consulta; se rellenaron por "
                         f"persistencia. Los horizontes menores a {min_h} días son poco confiables.")
        if stale > self.spec["stale_fill_days"] + 3:
            notes.append("El retraso supera lo evaluado; el error puede ser mayor al medido.")
        return Forecast(region_used=region, last_observed=last.date().isoformat(), as_of=as_of_ts.date().isoformat(),
                        stale_days=stale, days=days, values=values, spread=spread, source=source,
                        degraded=degraded, notes=notes)

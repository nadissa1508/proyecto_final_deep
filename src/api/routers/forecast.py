"""Endpoint del módulo de pronóstico: región → clima de los próximos 14 días.

`GET /api/forecast/{region}`. La historia reciente viene de Open-Meteo (ERA5, que llega con unos 5–8 días de retraso); los días que
faltan se rellenan por persistencia y los horizontes menores a `min_horizon_with_gap` se marcan como poco confiables. Con
`source=local` se usa el CSV descargado (útil sin internet); si la consulta en vivo falla, se cae a local y se avisa.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.api.state import get_state
from src.forecast.download import load_regions

router = APIRouter(prefix="/api/forecast", tags=["forecast"])


class ForecastResponse(BaseModel):
    region: str
    fecha_consulta: str
    ultimo_dia_observado: str
    dias_de_retraso: int
    fuente_historia: str
    dias: List[str]
    valores: Dict[str, List[float]]
    dispersion_entre_modelos: Dict[str, List[float]]
    poco_confiable: List[bool]
    notas: List[str]


def load_history(region: str, source: str):
    """Devuelve (historia, fuente, notas). `live` consulta Open-Meteo; si falla cae al CSV local."""
    from src.forecast.data import load_region
    from src.forecast.infer import fetch_history
    spec = load_regions()[region]
    if source == "live":
        try:
            return fetch_history(spec["lat"], spec["lon"], spec["elevation_m"]), "Open-Meteo (ERA5)", []
        except Exception as exc:                                   # sin internet, límite de peticiones, etc.
            note = f"No se pudo consultar Open-Meteo ({exc.__class__.__name__}); se usó la historia descargada."
            return load_region(region), "archivo local", [note]
    return load_region(region), "archivo local", []


@router.get("/{region}", response_model=ForecastResponse)
def forecast(region: str, source: str = Query("live", pattern="^(live|local)$"),
             as_of: date | None = Query(None, description="fecha de consulta; por defecto hoy")):
    forecaster = get_state().forecaster
    if forecaster is None:
        raise HTTPException(503, f"Módulo de pronóstico no disponible: {get_state().forecast_error}")
    if region not in load_regions():
        raise HTTPException(404, f"Región desconocida: {region}. Disponibles: {', '.join(load_regions())}")
    history, origin, notes = load_history(region, source)
    try:  # historia inválida o as_of anterior a la historia: error del pedido (422), no del servidor (500)
        result = forecaster.predict(history, region, as_of=as_of or date.today())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if result.stale_days > 30:
        raise HTTPException(422, f"La historia termina el {result.last_observed}, {result.stale_days} días antes de la consulta")
    return ForecastResponse(
        region=region, fecha_consulta=result.as_of, ultimo_dia_observado=result.last_observed,
        dias_de_retraso=result.stale_days, fuente_historia=origin, dias=result.days, valores=result.values,
        dispersion_entre_modelos=result.spread, poco_confiable=result.degraded, notas=notes + result.notes)

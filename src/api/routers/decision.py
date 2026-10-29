"""Endpoint de decisión: región + fotos opcionales → creencia sobre la roya y acción recomendada.

`POST /api/decision` (multipart/form-data). Une el pronóstico de la región (evidencia climática para la red bayesiana), las
hojas fotografiadas (observaciones del clasificador) y el MDP (política QMDP con la creencia). Sin fotos decide solo con el clima.
Responde 503 si falta algún módulo. La acción es una recomendación de apoyo, no una orden; la incidencia que se reporta es la de las
hojas fotografiadas, con su intervalo, y no reemplaza el muestreo de ANACAFE.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.api.routers.forecast import load_history
from src.api.state import get_state
from src.api.uploads import read_images
from src.forecast.download import load_regions
from src.fusion.contracts import STATE_NAMES
from src.fusion.pipeline import leaf_from_prediction
from src.mdp.environment import ParcelaProfile

router = APIRouter(prefix="/api/decision", tags=["decision"])


class IncidenceOut(BaseModel):
    hojas: int
    hojas_con_roya: int
    aparente: float
    corregida: float
    intervalo_95: List[float]
    sensibilidad: float
    especificidad: float


class ClimateOut(BaseModel):
    ventana_dias: int
    mes: int
    temperatura_media: float
    humedad_media: float
    lluvia_total_mm: float
    respecto_de_lo_habitual: Dict[str, Dict[str, float]]
    poco_confiable: bool


class DecisionResponse(BaseModel):
    region: str
    accion: str
    valor_esperado_por_accion: Dict[str, float]
    creencia_solo_clima: Dict[str, float]
    creencia: Dict[str, float]
    hojas: List[str]
    incidencia: Optional[IncidenceOut] = None
    clima: ClimateOut
    notas: List[str]
    advertencia: str


def _dist(values) -> Dict[str, float]:
    return {name: round(float(v), 4) for name, v in zip(STATE_NAMES, values)}


@router.post("", response_model=DecisionResponse)
async def decide(region: str = Form(...), files: List[UploadFile] = File(default=[]), source: str = Form("live"),
                 altitud: Optional[float] = Form(None, ge=0, le=4000), sombra: int = Form(1, ge=0, le=2),
                 humedad_tipica: Optional[float] = Form(None, ge=0, le=100)):
    state = get_state()
    if region not in load_regions():
        raise HTTPException(404, f"Región desconocida: {region}. Disponibles: {', '.join(load_regions())}")
    if source not in ("live", "local"):
        raise HTTPException(422, "source debe ser 'live' o 'local'")
    if state.forecaster is None or state.fusion is None:
        raise HTTPException(503, f"Módulo no disponible: {state.forecast_error or state.fusion_error}")
    if files and state.vision is None:
        raise HTTPException(503, f"Módulo de visión no disponible: {state.vision_error}")

    leaves = []
    if files:
        images = await read_images(files)
        leaves = [leaf_from_prediction(p) for p in state.vision.predict(images)]
    history, _, notes = load_history(region, source)
    forecast = state.forecaster.predict(history, region, as_of=date.today())
    if forecast.stale_days > 30:
        raise HTTPException(422, f"La historia termina el {forecast.last_observed}, {forecast.stale_days} días antes de hoy")

    spec = load_regions()[region]
    profile = ParcelaProfile(altitud=float(altitud if altitud is not None else spec["elevation_m"]), sombra=int(sombra),
                             humedad_tipica=float(humedad_tipica if humedad_tipica is not None
                                                  else history["relative_humidity_2m_mean"].tail(90).mean()))
    if humedad_tipica is None:
        notes.append("Humedad típica de la parcela estimada con la humedad de los últimos 90 días.")
    climate = state.fusion.climate(forecast)
    decision = state.fusion.decide(climate.window.month, climate, leaves, profile)
    w = climate.window
    inc = decision.belief.incidence
    return DecisionResponse(
        region=region, accion=decision.action_name, valor_esperado_por_accion=decision.expected_value,
        creencia_solo_clima=_dist(decision.belief.climate_only), creencia=_dist(decision.belief.posterior),
        hojas=[["sana", "roya_leve", "roya_severa", "otra_afeccion"][leaf.label] for leaf in leaves],
        incidencia=None if inc is None else IncidenceOut(
            hojas=inc.n_leaves, hojas_con_roya=inc.n_rust_predicted, aparente=inc.apparent, corregida=inc.corrected,
            intervalo_95=[inc.low, inc.high], sensibilidad=inc.sensitivity, especificidad=inc.specificity),
        clima=ClimateOut(ventana_dias=w.horizon_days, mes=w.month, temperatura_media=w.mean_temperature,
                         humedad_media=w.mean_humidity, lluvia_total_mm=w.total_rain,
                         respecto_de_lo_habitual={"temperatura": _dist(climate.p_T), "humedad": _dist(climate.p_H),
                                                  "lluvia": _dist(climate.p_P)}, poco_confiable=w.degraded),
        notas=notes + forecast.notes + decision.notes,
        advertencia="Recomendación de apoyo basada en un modelo con supuestos documentados (docs/decision_umbral_roya.md); "
                    "no reemplaza el muestreo de campo ni la asesoría técnica.")

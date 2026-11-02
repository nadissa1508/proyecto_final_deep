"""Endpoint de decisión: región + fotos opcionales → creencia sobre la roya y acción recomendada.

`POST /api/decision` (multipart/form-data). Une el pronóstico de la región (evidencia climática para la red bayesiana), las
hojas fotografiadas (observaciones del clasificador) y el MDP (política QMDP con la creencia). Sin fotos decide solo con el clima.
Responde 503 si falta algún módulo. La acción es una recomendación de apoyo, no una orden; la incidencia que se reporta es la de las
hojas fotografiadas, con su intervalo, y no reemplaza el muestreo de ANACAFE.

`as_of` (opcional, uso académico) hace una **consulta histórica**, igual que en el pronóstico (`history_as_of`): sirve para ver cómo responde
el sistema en otra época del año (por ejemplo, en la época de lluvias); no es un pronóstico de hoy.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.api.routers.forecast import history_as_of
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


class AlertOut(BaseModel):
    nivel: str = Field(..., description="seguimiento | vigilancia | sospecha | alerta")
    roya: str = Field(..., description="detectada | sospecha | no_detectada | sin_fotos (solo de las fotos)")
    riesgo_climatico: str = Field(..., description="alto | moderado | bajo (pronóstico y época)")
    mensaje: str
    discrepancia_mdp: Optional[str] = None
    dias_favorables: int
    ventana_dias: int
    hojas_con_roya: int
    hojas: int
    epoca_de_lluvias: bool
    poco_confiable: bool
    razones: List[str]


class DecisionResponse(BaseModel):
    region: str
    accion: str
    valor_esperado_por_accion: Dict[str, float]
    creencia_solo_clima: Dict[str, float]
    creencia: Dict[str, float]
    hojas: List[str]
    hojas_descartadas: int = 0
    alarma: Optional[AlertOut] = None
    incidencia: Optional[IncidenceOut] = None
    clima: ClimateOut
    fecha_consulta: str
    consulta_historica: bool = False
    notas: List[str]
    advertencia: str


def _dist(values) -> Dict[str, float]:
    return {name: round(float(v), 4) for name, v in zip(STATE_NAMES, values)}


@router.post("", response_model=DecisionResponse)
async def decide(region: str = Form(...), files: List[UploadFile] = File(default=[]), source: str = Form("live"),
                 altitud: Optional[float] = Form(None, ge=0, le=4000), sombra: int = Form(1, ge=0, le=2),
                 humedad_tipica: Optional[float] = Form(None, ge=0, le=100), as_of: Optional[date] = Form(None)):
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
    discarded: List[str] = []
    if files:
        images = await read_images(files)
        for upload, prediction in zip(files, state.vision.predict(images)):
            if getattr(prediction, "usable", True):
                leaves.append(leaf_from_prediction(prediction))
            else:
                discarded.append(f"{upload.filename or 'foto'}: {prediction.reason}")
    history, _, notes = history_as_of(region, source, as_of)
    if discarded:
        notes.append(f"{len(discarded)} de {len(files)} fotos no se usaron por no ser concluyentes ({'; '.join(discarded[:5])}"
                     f"{'…' if len(discarded) > 5 else ''}). Tome la foto de una hoja completa, con buena luz y sin movimiento.")
    try:  # historia insuficiente para la fecha pedida: error del pedido (422), no del servidor (500)
        forecast = state.forecaster.predict(history, region, as_of=as_of or date.today())
    except ValueError as exc:
        raise HTTPException(422, f"No se pudo pronosticar desde {as_of or date.today()}: {exc}") from exc
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
    alert = state.fusion.alert(forecast, leaves)
    discrepancy = None
    if alert.rust == "detectada" and decision.action_name == "nada":
        discrepancy = ("Hay roya detectada, pero el MDP no recomienda aplicar este mes (por el costo, la época y el nivel estimado). "
                       "No lo tome como un descarte: evalúe la parcela con un técnico.")
    w = climate.window
    inc = decision.belief.incidence
    return DecisionResponse(
        region=region, accion=decision.action_name, valor_esperado_por_accion=decision.expected_value,
        creencia_solo_clima=_dist(decision.belief.climate_only), creencia=_dist(decision.belief.posterior),
        hojas=[["sana", "roya_leve", "roya_severa", "otra_afeccion"][leaf.label] for leaf in leaves],
        hojas_descartadas=len(discarded),
        alarma=AlertOut(nivel=alert.level, roya=alert.rust, riesgo_climatico=alert.risk, mensaje=alert.message,
                        discrepancia_mdp=discrepancy, dias_favorables=alert.favorable_days, ventana_dias=alert.window_days,
                        hojas_con_roya=alert.rust_leaves, hojas=alert.n_leaves, epoca_de_lluvias=alert.in_season,
                        poco_confiable=alert.unreliable, razones=alert.reasons),
        incidencia=None if inc is None else IncidenceOut(
            hojas=inc.n_leaves, hojas_con_roya=inc.n_rust_predicted, aparente=inc.apparent, corregida=inc.corrected,
            intervalo_95=[inc.low, inc.high], sensibilidad=inc.sensitivity, especificidad=inc.specificity),
        clima=ClimateOut(ventana_dias=w.horizon_days, mes=w.month, temperatura_media=w.mean_temperature,
                         humedad_media=w.mean_humidity, lluvia_total_mm=w.total_rain,
                         respecto_de_lo_habitual={"temperatura": _dist(climate.p_T), "humedad": _dist(climate.p_H),
                                                  "lluvia": _dist(climate.p_P)}, poco_confiable=w.degraded),
        fecha_consulta=forecast.as_of, consulta_historica=as_of is not None,
        notas=notes + forecast.notes + decision.notes,
        advertencia="Recomendación de apoyo basada en un modelo con supuestos documentados (docs/decision_umbral_roya.md); "
                    "no reemplaza el muestreo de campo ni la asesoría técnica.")

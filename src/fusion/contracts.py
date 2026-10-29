"""Tipos de intercambio entre los módulos de visión, pronóstico, red bayesiana y MDP.

`vision` y `forecast` no se importan entre sí: se unen únicamente aquí, a través de estos tipos. Los adaptadores que
convierten `LeafPrediction` y `Forecast` en estos tipos viven en `src/fusion/` (nunca en los módulos de origen).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

STATE_NAMES = ("bajo", "medio", "alto")           # T, H, P (respecto de lo habitual) y R (nivel de roya de la parcela)
LEAF_CLASSES = ("sana", "roya_leve", "roya_severa", "otra_afeccion")


@dataclass(frozen=True)
class LeafObservation:
    """Lo que el clasificador devuelve para una hoja (clase predicha, ya calibrada y en ensamble)."""
    label: int
    probs: Tuple[float, ...] = ()
    agree: bool = True


@dataclass(frozen=True)
class ClimateWindow:
    """Resumen de la ventana pronosticada: lo que ve la red bayesiana en vez de un valor puntual."""
    region: str
    month: int                       # mes calendario del primer día pronosticado
    horizon_days: int
    mean_temperature: float          # °C, media de las temperaturas medias diarias
    mean_humidity: float             # %, media de las humedades relativas diarias
    total_rain: float                # mm acumulados en la ventana
    stale_days: int = 0              # retraso de ERA5 rellenado por persistencia
    degraded: bool = False           # la ventana incluye horizontes marcados como poco confiables


@dataclass(frozen=True)
class ClimateEvidence:
    """Evidencia blanda: distribución sobre (bajo, medio, alto) de T, H y P respecto de lo habitual."""
    window: ClimateWindow
    p_T: Tuple[float, float, float]
    p_H: Tuple[float, float, float]
    p_P: Tuple[float, float, float]


@dataclass(frozen=True)
class IncidenceEstimate:
    """Incidencia de la parcela estimada con las hojas fotografiadas."""
    n_leaves: int
    n_rust_predicted: int
    apparent: float                  # fracción de hojas con roya predicha
    corrected: float                 # Rogan–Gladen, usando sensibilidad y especificidad medidas
    low: float                       # intervalo de Wilson (95 %) propagado por la corrección
    high: float
    sensitivity: float
    specificity: float


@dataclass
class Belief:
    """Creencia sobre el nivel de roya de la parcela R, con y sin la evidencia de las fotos."""
    climate_only: Tuple[float, float, float]          # P(R | clima)
    posterior: Tuple[float, float, float]             # P(R | clima, hojas)
    n_leaves: int = 0
    incidence: Optional[IncidenceEstimate] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class Decision:
    month: int
    action: int
    action_name: str
    expected_value: Dict[str, float]                  # Σ_s b(s) Q*(s, a) por acción
    belief: Belief
    climate: Optional[ClimateEvidence] = None
    notes: List[str] = field(default_factory=list)

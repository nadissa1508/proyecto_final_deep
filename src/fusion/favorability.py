"""Alerta de roya y riesgo climático: dos salidas independientes que se combinan en un mensaje.

Revisión de la regla C de `docs/decision_umbral_roya.md` (§1 y §11). La versión anterior exigía roya + clima favorable + época para dar la alarma,
así que una detección con clima no favorable bajaba a «vigilancia». Ahora:

1. **Roya** (solo de las fotos, nunca depende del clima): `detectada` (al menos `min_rust_leaves` hojas con roya predicha), `sospecha` (alguna hoja,
   menos que el mínimo: una sola puede ser un falso positivo), `no_detectada` o `sin_fotos`.
2. **Riesgo climático** (del pronóstico y la época): `alto` (al menos `min_favorable_days` de los próximos `window_days` días favorables en época de
   lluvias), `moderado` (días favorables fuera de la época, o algún día favorable en la época) o `bajo`. `poco_confiable` marca aparte que el pronóstico
   se hizo con una historia de entrada completada por persistencia (retraso de ERA5); no se convierte en un nivel, porque en operación siempre ocurre.
3. **Mensaje** (`level`): con roya detectada siempre hay `alerta` (el riesgo solo cambia la urgencia); `sospecha` si hay una hoja dudosa; `vigilancia`
   si no hay roya pero el riesgo es alto; `seguimiento` en el resto.

Día favorable = temperatura media dentro del rango del perfil y humedad relativa media mayor a 85 % (CEDICAFE/ANACAFE; la hoja mojada y la oscuridad no
se modelan). El perfil `altura` (18–25 °C) y todos los mínimos son supuestos de `configs/fusion.yaml`. **La alerta no ordena aplicar fungicida: la acción la
sigue decidiendo el MDP**, y cuando no coinciden se avisa (`decision.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from src.fusion.config import load_config

RUST_LABELS = (1, 2)  # roya_leve, roya_severa
LEVELS = ("seguimiento", "vigilancia", "sospecha", "alerta")
MESSAGES = {
    ("alerta", "alto"): "Roya detectada y alto riesgo de propagación: evalúe una intervención preventiva pronto.",
    ("alerta", "moderado"): "Roya detectada: haga seguimiento fitosanitario y evalúe el manejo preventivo; el clima no lo descarta.",
    ("alerta", "bajo"): "Roya detectada: haga seguimiento fitosanitario; aunque el clima sea poco favorable, no descarte el manejo preventivo.",
    "sospecha": "Posible roya en una hoja: fotografíe más hojas de plantas distintas para confirmarlo.",
    "vigilancia": "Sin roya detectada, pero con condiciones favorables en época de lluvias: vigile la parcela.",
    "seguimiento": "Sin roya detectada ni riesgo climático alto: seguimiento normal.",
}


@dataclass(frozen=True)
class FavorabilityConfig:
    t_lo: float
    t_hi: float
    humidity_min: float
    window_days: int
    min_favorable_days: int
    season_months: tuple
    min_rust_leaves: int
    profile: str = ""

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None, profile: Optional[str] = None) -> "FavorabilityConfig":
        f = (cfg or load_config())["favorability"]
        name = profile or f["profile"]
        lo, hi = f["profiles"][name]
        return cls(t_lo=float(lo), t_hi=float(hi), humidity_min=float(f["humidity_min"]), window_days=int(f["window_days"]),
                   min_favorable_days=int(f["min_favorable_days"]), season_months=tuple(int(m) for m in f["season_months"]),
                   min_rust_leaves=int(f["min_rust_leaves"]), profile=name)


def favorable_days(temperature: Sequence[float], humidity: Sequence[float], cfg: FavorabilityConfig) -> np.ndarray:
    """Máscara de días favorables: temperatura media en el rango del perfil y humedad media mayor al mínimo."""
    t, h = np.asarray(temperature, dtype=float), np.asarray(humidity, dtype=float)
    return (t >= cfg.t_lo) & (t <= cfg.t_hi) & (h > cfg.humidity_min)


def window_event(temperature: Sequence[float], humidity: Sequence[float], cfg: FavorabilityConfig) -> bool:
    """¿Hay al menos `min_favorable_days` días favorables en la ventana de `window_days` días?"""
    mask = favorable_days(np.asarray(temperature)[: cfg.window_days], np.asarray(humidity)[: cfg.window_days], cfg)
    return int(mask.sum()) >= cfg.min_favorable_days


def rust_status(leaf_labels: Sequence[int], cfg: FavorabilityConfig) -> str:
    if not leaf_labels:
        return "sin_fotos"
    rust = sum(1 for lbl in leaf_labels if lbl in RUST_LABELS)
    if rust >= cfg.min_rust_leaves:
        return "detectada"
    return "sospecha" if rust > 0 else "no_detectada"


def climate_risk(n_favorable: int, in_season: bool, cfg: FavorabilityConfig) -> str:
    if n_favorable >= cfg.min_favorable_days and in_season:
        return "alto"
    if n_favorable >= cfg.min_favorable_days or (in_season and n_favorable > 0):
        return "moderado"
    return "bajo"


@dataclass
class PreventiveAlert:
    level: str                      # seguimiento | vigilancia | sospecha | alerta
    rust: str                       # detectada | sospecha | no_detectada | sin_fotos
    risk: str                       # alto | moderado | bajo
    message: str
    favorable_days: int
    window_days: int
    rust_leaves: int
    n_leaves: int
    in_season: bool
    unreliable: bool                # la historia de entrada se completó por persistencia (retraso de ERA5) y hay horizontes marcados como poco confiables
    reasons: List[str] = field(default_factory=list)


def preventive_alert(temperature: Sequence[float], humidity: Sequence[float], month: int, leaf_labels: Sequence[int],
                     cfg: FavorabilityConfig, unreliable: bool = False) -> PreventiveAlert:
    window = cfg.window_days
    n_fav = int(favorable_days(np.asarray(temperature)[:window], np.asarray(humidity)[:window], cfg).sum())
    in_season = month in cfg.season_months
    rust = rust_status(leaf_labels, cfg)
    risk = climate_risk(n_fav, in_season, cfg)
    if rust == "detectada":
        level, message = "alerta", MESSAGES[("alerta", risk)]
    elif rust == "sospecha":
        level, message = "sospecha", MESSAGES["sospecha"]
    elif risk == "alto":
        level, message = "vigilancia", MESSAGES["vigilancia"]
    else:
        level, message = "seguimiento", MESSAGES["seguimiento"]
    n_rust = sum(1 for lbl in leaf_labels if lbl in RUST_LABELS)
    reasons = [f"{n_rust} de {len(leaf_labels)} hojas con roya" if leaf_labels else "sin fotos: no se puede saber si hay roya",
               f"{n_fav} de los próximos {window} días con condiciones favorables ({cfg.t_lo:g}–{cfg.t_hi:g} °C y humedad > {cfg.humidity_min:g} %)",
               "época de lluvias" if in_season else "fuera de la época de lluvias"]
    if unreliable:
        reasons.append("el pronóstico se hizo con datos recientes incompletos (retraso de ERA5): el riesgo climático es orientativo")
    return PreventiveAlert(level, rust, risk, message, n_fav, window, n_rust, len(leaf_labels), in_season, unreliable, reasons)

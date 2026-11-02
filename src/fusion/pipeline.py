"""Motor de decisión: une pronóstico, fotos, red bayesiana extendida y QMDP. Es lo que usa la API.

Recibe objetos con la forma de `forecast.infer.Forecast` y `vision.infer.LeafPrediction`, pero solo lee sus campos (no importa
esos módulos), de modo que `vision` y `forecast` siguen sin conocerse entre sí.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from src.fusion.config import load_config
from src.fusion.contracts import Belief, ClimateEvidence, Decision, LeafObservation
from src.fusion.evidence import EVIDENCE_PATH, EvidenceTable, climate_evidence, window_from_values
from src.fusion.favorability import FavorabilityConfig, PreventiveAlert, preventive_alert
from src.fusion.network_ext import ExtendedModel
from src.fusion.qmdp import qmdp, solve_parcel
from src.mdp.environment import ACTION_NAMES, ParcelaProfile


def leaf_from_prediction(prediction) -> LeafObservation:
    """Adapta un `LeafPrediction` del módulo de visión."""
    return LeafObservation(label=int(prediction.label), probs=tuple(prediction.probs), agree=bool(prediction.agree))


class FusionEngine:
    def __init__(self, cfg: Optional[dict] = None, table: Optional[EvidenceTable] = None) -> None:
        self.cfg = cfg or load_config()
        self.model = ExtendedModel.build(self.cfg)
        self.table = table or EvidenceTable.load(EVIDENCE_PATH)
        self.favorability = FavorabilityConfig.from_config(self.cfg)
        self._q: Dict[Tuple, np.ndarray] = {}

    def q_values(self, profile: Optional[ParcelaProfile] = None) -> np.ndarray:
        key = None if profile is None else (profile.altitud, profile.sombra, profile.humedad_tipica)
        if key not in self._q:
            self._q[key] = solve_parcel(profile).Q
        return self._q[key]

    def climate(self, forecast, horizon: Optional[int] = None) -> ClimateEvidence:
        horizon = horizon or self.cfg["decision_horizon"]
        window = window_from_values(forecast.values, horizon, forecast.region_used,
                                    month=int(forecast.days[0][5:7]), stale_days=forecast.stale_days,
                                    degraded=forecast.degraded)
        return climate_evidence(window, self.table)

    def alert(self, forecast, leaves: Sequence[LeafObservation] = ()) -> PreventiveAlert:
        """Alerta de roya (solo de las fotos) y riesgo climático (días favorables pronosticados y época); ver decision_umbral_roya.md §11."""
        window = self.favorability.window_days
        return preventive_alert(forecast.values["temperature_2m_mean"], forecast.values["relative_humidity_2m_mean"],
                                month=int(forecast.days[0][5:7]), leaf_labels=[leaf.label for leaf in leaves], cfg=self.favorability,
                                unreliable=any(forecast.degraded[:window]))

    def decide(self, month: int, climate: Optional[ClimateEvidence], leaves: Sequence[LeafObservation] = (),
               profile: Optional[ParcelaProfile] = None) -> Decision:
        belief: Belief = self.model.belief(climate, [leaf.label for leaf in leaves])
        result = qmdp(belief.posterior, month, self.q_values(profile))
        notes = list(belief.notes)
        if not leaves:
            notes.append("Sin fotos, la decisión se apoya solo en el clima, que en las simulaciones aportó poco; "
                         "fotografíe hojas de varias plantas para una recomendación más fiable.")
        if leaves and any(not leaf.agree for leaf in leaves):
            notes.append("En algunas hojas los modelos de visión no coinciden; se usó la clase del ensamble.")
        return Decision(month=month, action=result.action, action_name=result.action_name,
                        expected_value={ACTION_NAMES[a]: float(v) for a, v in enumerate(result.values)},
                        belief=belief, climate=climate, notes=notes)

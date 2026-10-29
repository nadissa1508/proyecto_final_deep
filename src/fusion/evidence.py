"""Pronóstico → evidencia climática (T, H, P) para la red bayesiana.

La red heredada discretiza con cortes absolutos que no sirven aquí: la lluvia estaba en mm anuales (1,200–2,000) y la
CPT de R solo tenía datos en 4 de sus 27 combinaciones (el resto quedaba uniforme). Aquí cada variable de la ventana
pronosticada se compara con **lo habitual en esa finca y esa época**:

- La ventana (7 o 14 días) se resume como temperatura media, humedad media y lluvia acumulada.
- `bajo / medio / alto` son los terciles de esa misma cantidad en el historial de pronósticos del modelo para la región
  y el mes (`build_evidence.py`, 1980–2017). Se usan terciles **de los pronósticos y no de ERA5** para que el modelo se
  compare consigo mismo (el modelo afinado de lluvia y humedad predice lo que mide la estación, que difiere de ERA5).
- El pronóstico tiene error: en vez de asignar un solo estado se reparte la probabilidad con una normal centrada en el
  valor pronosticado y desviación igual al error típico de la ventana (medido en 2018–2020 y guardado en la tabla).
  La lluvia acumulada es positiva y muy sesgada, así que su normal se plantea en `log1p(mm)` (cortes y desviación incluidos);
  como `log1p` es monótona, los terciles son los mismos. Es evidencia blanda ("virtual evidence"): entra a la red como el
  valor de los factores T, H y P.

Por qué los terciles son relativos: en estas fincas la temperatura media es de 17–21 °C, por debajo del rango de 21–25 °C que
CEDICAFE señala como favorable; un corte absoluto dejaría T siempre en "bajo". Un tercil "alto" significa "más cálido de lo
habitual para esta finca y época", que es lo que aumenta el riesgo. El nivel de base de cada finca (altitud, humedad típica)
lo aporta el MDP con `rust_pressure`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

from src.common.paths import MODELS_DIR
from src.fusion.contracts import ClimateEvidence, ClimateWindow

EVIDENCE_PATH = MODELS_DIR / "fusion" / "evidence.json"
TEMPERATURE, HUMIDITY, RAIN = "temperature_2m_mean", "relative_humidity_2m_mean", "precipitation_sum"
VARS = {"T": TEMPERATURE, "H": HUMIDITY, "P": RAIN}


class EvidenceNotAvailable(RuntimeError):
    """No existe la tabla de terciles: hay que ejecutar `python -m src.fusion.build_evidence`."""


@dataclass
class EvidenceTable:
    thresholds: Dict[str, Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]]    # región → horizonte → mes → var → (q33, q67)
    sigma: Dict[str, Dict[str, float]]                                              # horizonte → var → desviación típica
    meta: dict

    @classmethod
    def load(cls, path: Path = EVIDENCE_PATH) -> "EvidenceTable":
        if not path.exists():
            raise EvidenceNotAvailable(f"No existe la tabla de evidencia: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(raw["thresholds"], raw["sigma"], raw.get("meta", {}))

    @property
    def regions(self) -> Sequence[str]:
        return list(self.thresholds)

    def cuts(self, region: str, horizon: int, month: int) -> Dict[str, Tuple[float, float]]:
        try:
            return self.thresholds[region][str(horizon)][str(month)]
        except KeyError as exc:
            raise ValueError(f"Sin terciles para región={region}, horizonte={horizon}, mes={month}") from exc


def window_from_values(values: Mapping[str, Sequence[float]], horizon: int, region: str, month: int,
                       stale_days: int = 0, degraded: Sequence[bool] = ()) -> ClimateWindow:
    """Resume los primeros `horizon` días de un pronóstico diario en la ventana que ve la red."""
    if horizon > len(values[TEMPERATURE]):
        raise ValueError(f"El pronóstico tiene {len(values[TEMPERATURE])} días, se pidieron {horizon}")
    t, h, p = (values[TEMPERATURE][:horizon], values[HUMIDITY][:horizon], values[RAIN][:horizon])
    return ClimateWindow(region=region, month=month, horizon_days=horizon,
                         mean_temperature=sum(t) / horizon, mean_humidity=sum(h) / horizon, total_rain=float(sum(p)),
                         stale_days=stale_days, degraded=any(degraded[:horizon]) if degraded else False)


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def tercile_probabilities(value: float, cuts: Tuple[float, float], sigma: float) -> Tuple[float, float, float]:
    """P(bajo), P(medio), P(alto) si el valor real es normal(value, sigma) y los cortes son (q33, q67)."""
    lo, hi = cuts
    if sigma <= 0:
        return (1.0, 0.0, 0.0) if value < lo else ((0.0, 1.0, 0.0) if value < hi else (0.0, 0.0, 1.0))
    p_low = _cdf((lo - value) / sigma)
    p_le_mid = _cdf((hi - value) / sigma)
    return (p_low, p_le_mid - p_low, 1.0 - p_le_mid)


def climate_evidence(window: ClimateWindow, table: EvidenceTable) -> ClimateEvidence:
    cuts = table.cuts(window.region, window.horizon_days, window.month)
    sigma = table.sigma[str(window.horizon_days)]
    value = {"T": window.mean_temperature, "H": window.mean_humidity, "P": math.log1p(max(window.total_rain, 0.0))}
    cuts = {**cuts, "P": tuple(math.log1p(max(c, 0.0)) for c in cuts["P"])}        # la lluvia se evalúa en log1p(mm)
    probs = {k: tercile_probabilities(value[k], cuts[k], sigma[k]) for k in VARS}
    return ClimateEvidence(window=window, p_T=probs["T"], p_H=probs["H"], p_P=probs["P"])

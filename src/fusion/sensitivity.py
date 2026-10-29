"""Sensibilidad de la decisión a los supuestos de diseño de `configs/fusion.yaml`.

Los cortes de roya, la mezcla de hojas y la CPT climática no salen de datos de campo. Aquí se mueve cada supuesto, uno a la vez, y se
mide cuánto cambian la creencia y la acción frente a la configuración base, sobre una malla fija de casos:

- clima: desfavorable, neutro y favorable (evidencia blanda sobre T, H y P respecto de lo habitual);
- fotos: 0, 10 y 30 hojas, con distintas proporciones de hojas con roya predicha;
- mes: marzo (seca), julio y octubre (lluvia, cosecha cerca).

Salidas: `experiments/fusion/sensibilidad.csv` y una tabla resumen. Un supuesto "importa" si cambia la acción en una fracción grande
de casos; si no, la conclusión no depende de él.

Uso:
    python -m src.fusion.sensitivity
"""

from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np
import pandas as pd

from src.common.paths import EXPERIMENTS_DIR
from src.fusion.config import load_config
from src.fusion.contracts import ClimateEvidence, ClimateWindow
from src.fusion.network_ext import ExtendedModel
from src.fusion.qmdp import qmdp, solve_parcel

WINDOW = ClimateWindow(region="-", month=1, horizon_days=14, mean_temperature=0.0, mean_humidity=0.0, total_rain=0.0)
CLIMATES = {
    "desfavorable": ((0.8, 0.15, 0.05), (0.8, 0.15, 0.05), (0.8, 0.15, 0.05)),
    "neutro": ((1 / 3,) * 3, (1 / 3,) * 3, (1 / 3,) * 3),
    "favorable": ((0.05, 0.15, 0.8), (0.05, 0.15, 0.8), (0.05, 0.15, 0.8)),
}
MONTHS = (3, 7, 10)
LEAVES = [(n, k) for n in (0, 10, 30) for k in ((0,) if n == 0 else (0, n // 10, n // 3, n))]      # k hojas con roya de n

# (nombre, anulaciones de la configuración)
VARIATIONS = [
    ("base", {}),
    ("cortes 3 % / 8 % (incidencia 1 / 5 / 12 %)", {"rust_bands": {"representative_incidence": [0.01, 0.05, 0.12]}}),
    ("cortes 7 % / 15 % (incidencia 3 / 10 / 20 %)", {"rust_bands": {"representative_incidence": [0.03, 0.10, 0.20]}}),
    ("proporción severa 0.1", {"leaf_mix": {"severe_share": 0.1}}),
    ("proporción severa 0.5", {"leaf_mix": {"severe_share": 0.5}}),
    ("otra afección 0.05", {"leaf_mix": {"other_share": 0.05}}),
    ("otra afección 0.4", {"leaf_mix": {"other_share": 0.4}}),
    ("clima neutro: P(R≥medio) 0.25", {"climate_risk": {"neutral_at_least_medium": 0.25, "neutral_high": 0.08}}),
    ("clima neutro: P(R≥medio) 0.65", {"climate_risk": {"neutral_at_least_medium": 0.65, "neutral_high": 0.30}}),
    ("pendiente del clima 0.25", {"climate_risk": {"slope": 0.25}}),
    ("pendiente del clima 1.0", {"climate_risk": {"slope": 1.0}}),
    ("matriz de confusión solo BRACOL", {"confusion": {"source": "bracol"}}),
    ("matriz de confusión solo RoCoLe", {"confusion": {"source": "rocole"}}),
    ("peso por hoja 0.5", {"leaf_weight": 0.5}),
    ("peso por hoja 0.25", {"leaf_weight": 0.25}),
]


def evaluate(overrides: dict, Q: np.ndarray) -> List[dict]:
    model = ExtendedModel.build(load_config(overrides))
    rows = []
    for (cname, probs), month, (n, k) in itertools.product(CLIMATES.items(), MONTHS, LEAVES):
        climate = ClimateEvidence(window=WINDOW, p_T=probs[0], p_H=probs[1], p_P=probs[2])
        labels = [1] * k + [0] * (n - k)                       # k hojas con roya (leve) y el resto sanas
        belief = model.belief(climate, labels)
        rows.append({"clima": cname, "mes": month, "hojas": n, "con_roya": k, "p_alto": belief.posterior[2],
                     "accion": qmdp(belief.posterior, month, Q).action_name})
    return rows


def main() -> None:
    Q = solve_parcel().Q
    base = pd.DataFrame(evaluate({}, Q))
    out, summary = [], []
    for name, overrides in VARIATIONS:
        table = pd.DataFrame(evaluate(overrides, Q)).assign(variacion=name)
        out.append(table)
        same = (table["accion"].to_numpy() == base["accion"].to_numpy())
        summary.append({"variacion": name, "casos": len(table), "accion_cambia": float(1 - same.mean()),
                        "cambio_medio_p_alto": float((table["p_alto"] - base["p_alto"]).abs().mean()),
                        "cambio_max_p_alto": float((table["p_alto"] - base["p_alto"]).abs().max())})
    EXPERIMENTS_DIR.joinpath("fusion").mkdir(exist_ok=True)
    pd.concat(out).round(4).to_csv(EXPERIMENTS_DIR / "fusion" / "sensibilidad.csv", index=False)
    result = pd.DataFrame(summary)
    result.round(4).to_csv(EXPERIMENTS_DIR / "fusion" / "sensibilidad_resumen.csv", index=False)
    pd.set_option("display.width", 200)
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Red bayesiana extendida: clima pronosticado y hojas fotografiadas → creencia sobre el nivel de roya de la parcela.

DAG (se reutiliza `BayesianNetwork` y Variable Elimination heredados, sin cambios):

    T, H, P → R → Y            (heredado: clima respecto de lo habitual → nivel de roya → rendimiento)
              R → O_1 ... O_n  (nuevo: lo que la CNN dijo de cada hoja fotografiada)

Dos niveles entre R y la foto, porque una foto describe una hoja y R describe la parcela:
`P(O | R) = Σ_L P(O | L) · P(L | R)`, con `P(O | L)` la matriz de confusión **medida** en el test de visión y `P(L | R)` la
mezcla de hojas que se espera en cada nivel (supuesto de `configs/fusion.yaml`). Cada hoja es una observación independiente dado
R (aproximación documentada: hojas de una misma planta no lo son), con un exponente `leaf_weight` para bajarles el peso.

Cambio frente a la red heredada: la CPT `P(R | T, H, P)` heredada se estimó con 100 filas de un conjunto de la India y solo tiene
datos en 4 de las 27 combinaciones (el suavizado de Laplace deja las otras 23 en 1/3, 1/3, 1/3), así que no responde al clima de
Guatemala. Aquí se reemplaza por un modelo ordinal explícito y parametrizado (supuesto de diseño con sensibilidad), no aprendido:
no existen datos de incidencia de roya emparejados con clima para estimarlo. `legacy_r_cpt` conserva la heredada para comparar.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.bayesian.cpt_estimator import YIELD_GIVEN_RUST
from src.bayesian.inference import variable_elimination
from src.bayesian.network import BayesianNetwork
from src.common.paths import EXPERIMENTS_DIR
from src.fusion.config import load_config
from src.fusion.contracts import STATE_NAMES, Belief, ClimateEvidence, IncidenceEstimate

CONFUSION_PATH = EXPERIMENTS_DIR / "vision" / "confusion_hoja.json"
N_STATE, N_LEAF = 3, 4
RUST_LABELS = (1, 2)                 # roya leve y severa


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def climate_risk_cpt(cfg: dict) -> np.ndarray:
    """P(R | T, H, P) con forma (R, T, H, P). Ordinal: z = índices T+H+P − 3; P(R≥medio) y P(R=alto) son logísticas en z."""
    cr = cfg["climate_risk"]
    a1, a2, b = _logit(cr["neutral_at_least_medium"]), _logit(cr["neutral_high"]), cr["slope"]
    if a2 > a1:
        raise ValueError("neutral_high no puede superar neutral_at_least_medium")
    cpt = np.zeros((N_STATE, N_STATE, N_STATE, N_STATE))
    for t in range(N_STATE):
        for h in range(N_STATE):
            for p in range(N_STATE):
                z = t + h + p - 3
                ge1, high = _sigmoid(a1 + b * z), _sigmoid(a2 + b * z)
                cpt[:, t, h, p] = [1.0 - ge1, ge1 - high, high]
    return cpt


def leaf_given_rust(cfg: dict) -> np.ndarray:
    """P(L | R) con forma (L, R): hoja ∈ {sana, roya leve, roya severa, otra afección} dado el nivel de la parcela."""
    inc = cfg["rust_bands"]["representative_incidence"]
    severe, other = cfg["leaf_mix"]["severe_share"], cfg["leaf_mix"]["other_share"]
    out = np.zeros((N_LEAF, N_STATE))
    for r in range(N_STATE):
        out[:, r] = [(1 - inc[r]) * (1 - other), inc[r] * (1 - severe), inc[r] * severe, (1 - inc[r]) * other]
    return out


def load_confusion_counts(source: str, path: Path = CONFUSION_PATH) -> np.ndarray:
    raw = json.loads(path.read_text(encoding="utf-8"))
    parts = {"bracol": ["test_bracol"], "rocole": ["test_rocole"], "pooled": ["test_bracol", "test_rocole"]}[source]
    return np.sum([np.array(raw[k], dtype=float) for k in parts], axis=0)


def obs_given_leaf(counts: np.ndarray, alpha: float) -> np.ndarray:
    """P(O | L) con forma (L, O): filas = clase real, suavizado de Laplace por celda."""
    smooth = counts + alpha
    return smooth / smooth.sum(axis=1, keepdims=True)


def sensitivity_specificity(counts: np.ndarray) -> Tuple[float, float]:
    """Detección de roya (leve o severa) a partir de la matriz de confusión: sensibilidad y especificidad."""
    rust, other = list(RUST_LABELS), [0, 3]
    sens = counts[np.ix_(rust, rust)].sum() / counts[rust].sum()
    spec = counts[np.ix_(other, other)].sum() / counts[other].sum()
    return float(sens), float(spec)


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def rogan_gladen(apparent: float, sens: float, spec: float) -> float:
    """Prevalencia corregida por los errores del clasificador: (aparente + esp − 1) / (sens + esp − 1), recortada a [0, 1]."""
    denom = sens + spec - 1.0
    return float(np.clip((apparent + spec - 1.0) / denom, 0.0, 1.0)) if denom > 1e-9 else float("nan")


def legacy_r_cpt(data_dir: Path) -> np.ndarray:
    """CPT P(R | T, H, P) heredada (forma (R, T, H, P)), para comparar con el escenario 'sistema anterior'."""
    import pandas as pd
    from src.bayesian.cpt_estimator import build_labeled_dataset, estimate_cpt
    raw = pd.read_csv(data_dir / "raw" / "crop_recommendation.csv")
    labeled = build_labeled_dataset(raw[raw["label"] == "coffee"].copy(), seed=42)
    card = {"T": 3, "H": 3, "P": 3, "R": 3, "Y": 3}
    return estimate_cpt(labeled, "R", ["T", "H", "P"], card)


@dataclass
class ExtendedModel:
    cfg: dict
    r_given_thp: np.ndarray            # (R, T, H, P)
    l_given_r: np.ndarray              # (L, R)
    o_given_l: np.ndarray              # (L, O)
    o_given_r: np.ndarray              # (O, R)
    y_given_r: np.ndarray              # (Y, R)
    sensitivity: float
    specificity: float

    @classmethod
    def build(cls, cfg: Optional[dict] = None, confusion_path: Path = CONFUSION_PATH,
              r_given_thp: Optional[np.ndarray] = None) -> "ExtendedModel":
        cfg = cfg or load_config()
        counts = load_confusion_counts(cfg["confusion"]["source"], confusion_path)
        o_given_l, l_given_r = obs_given_leaf(counts, cfg["confusion"]["alpha"]), leaf_given_rust(cfg)
        sens, spec = sensitivity_specificity(counts)
        return cls(cfg=cfg, r_given_thp=climate_risk_cpt(cfg) if r_given_thp is None else r_given_thp,
                   l_given_r=l_given_r, o_given_l=o_given_l, o_given_r=o_given_l.T @ l_given_r,
                   y_given_r=YIELD_GIVEN_RUST.T.copy(),               # (Y, R): las columnas suman 1
                   sensitivity=sens, specificity=spec)

    def network(self, climate: Optional[ClimateEvidence], n_leaves: int) -> BayesianNetwork:
        """Red con T, H, P como evidencia blanda (o uniforme si no hay pronóstico) y `n_leaves` nodos de observación."""
        uniform = np.full(N_STATE, 1.0 / N_STATE)
        pT, pH, pP = (np.array(climate.p_T), np.array(climate.p_H), np.array(climate.p_P)) if climate else (uniform,) * 3
        nodes = ["T", "H", "P", "R", "Y"] + [f"O{i}" for i in range(n_leaves)]
        parents: Dict[str, List[str]] = {"T": [], "H": [], "P": [], "R": ["T", "H", "P"], "Y": ["R"]}
        cpts = {"T": pT, "H": pH, "P": pP, "R": self.r_given_thp, "Y": self.y_given_r}
        card = {"T": 3, "H": 3, "P": 3, "R": 3, "Y": 3}
        for i in range(n_leaves):
            parents[f"O{i}"], cpts[f"O{i}"], card[f"O{i}"] = ["R"], self.o_given_r, N_LEAF
        return BayesianNetwork(nodes=nodes, parents=parents, cpts=cpts, cardinality=card,
                               states={"R": STATE_NAMES, "T": STATE_NAMES, "H": STATE_NAMES, "P": STATE_NAMES})

    def climate_posterior(self, climate: Optional[ClimateEvidence]) -> np.ndarray:
        """P(R | clima) por Variable Elimination (sin hojas)."""
        return variable_elimination(self.network(climate, 0), ["R"]).array

    def belief_exact(self, climate: Optional[ClimateEvidence], leaf_labels: Sequence[int]) -> np.ndarray:
        """P(R | clima, hojas) con una variable O_i por hoja dentro de la red (referencia para validar el camino rápido)."""
        net = self.network(climate, len(leaf_labels))
        return variable_elimination(net, ["R"], {f"O{i}": int(o) for i, o in enumerate(leaf_labels)}).array

    def incidence(self, leaf_labels: Sequence[int]) -> IncidenceEstimate:
        n = len(leaf_labels)
        k = sum(1 for o in leaf_labels if o in RUST_LABELS)
        low, high = wilson_interval(k, n)
        apparent = k / n if n else 0.0
        fix = lambda x: rogan_gladen(x, self.sensitivity, self.specificity)
        return IncidenceEstimate(n_leaves=n, n_rust_predicted=k, apparent=apparent, corrected=fix(apparent),
                                 low=fix(low), high=fix(high), sensitivity=self.sensitivity, specificity=self.specificity)

    def belief(self, climate: Optional[ClimateEvidence], leaf_labels: Sequence[int] = ()) -> Belief:
        """Creencia sobre R. Con `leaf_weight` = 1 coincide exactamente con `belief_exact`."""
        prior = self.climate_posterior(climate)
        w = float(self.cfg["leaf_weight"])
        post = prior.copy()
        for o in leaf_labels:
            post = post * self.o_given_r[int(o)] ** w
        post = post / post.sum()
        notes: List[str] = []
        n = len(leaf_labels)
        if 0 < n < self.cfg["min_leaves"]:
            notes.append(f"Hay {n} hojas; se recomiendan al menos {self.cfg['min_leaves']} de plantas distintas.")
        if climate is not None and climate.window.degraded:
            notes.append("El pronóstico incluye días poco confiables por el retraso de ERA5.")
        return Belief(climate_only=tuple(float(x) for x in prior), posterior=tuple(float(x) for x in post),
                      n_leaves=n, incidence=self.incidence(leaf_labels) if n else None, notes=notes)

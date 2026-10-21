"""Carga única de artefactos al arrancar la API.

Los cuatro módulos comparten artefactos pesados (varieties, CPTs, Champion,
política MDP) que se cargan una sola vez en el lifespan de FastAPI y se
guardan en este singleton. Los routers acceden vía `get_state()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.bayesian.cpt_estimator import build_labeled_dataset, estimate_cpt
from src.bayesian.network import BayesianNetwork, build_coffee_network
from src.csp.varieties_loader import Variety, load_varieties
from src.mdp.environment import build_environment
from src.mdp.solver import VISolution, value_iteration
from src.ml.models.champion import Champion
from src.ml.preprocess import MinMaxScaler, load_crop_dataset, stratified_split


@dataclass
class AppState:
    varieties: List[Variety] = field(default_factory=list)
    bayes_net: Optional[BayesianNetwork] = None
    ml_model: Optional[Champion] = None
    ml_scaler: Optional[MinMaxScaler] = None
    ml_classes: List[str] = field(default_factory=list)
    mdp_solution: Optional[VISolution] = None


_state: AppState = AppState()


def get_state() -> AppState:
    return _state


def load_all(data_dir: Path) -> AppState:
    _state.varieties = load_varieties(data_dir / "raw" / "varieties.csv")

    crop_path = data_dir / "raw" / "crop_recommendation.csv"
    raw = pd.read_csv(crop_path)
    coffee = raw[raw["label"] == "coffee"].copy()
    labeled = build_labeled_dataset(coffee, seed=42)
    card = {"T": 3, "H": 3, "P": 3, "R": 3, "Y": 3}
    T_cpt = estimate_cpt(labeled, "T", [], card)
    H_cpt = estimate_cpt(labeled, "H", [], card)
    P_cpt = estimate_cpt(labeled, "P", [], card)
    R_cpt = estimate_cpt(labeled, "R", ["T", "H", "P"], card)
    Y_cpt = estimate_cpt(labeled, "Y", ["R"], card)
    _state.bayes_net = build_coffee_network(T_cpt, H_cpt, P_cpt, R_cpt, Y_cpt)

    X, y, classes = load_crop_dataset(str(crop_path))
    X_tr, _, y_tr, _ = stratified_split(X, y, test_size=0.2, seed=42)
    scaler = MinMaxScaler().fit(X_tr)
    model = Champion().fit(scaler.transform(X_tr), y_tr)
    _state.ml_model = model
    _state.ml_scaler = scaler
    _state.ml_classes = classes

    T_mdp, R_mdp = build_environment()
    _state.mdp_solution = value_iteration(T_mdp, R_mdp, gamma=0.95, tol=1e-3)

    return _state

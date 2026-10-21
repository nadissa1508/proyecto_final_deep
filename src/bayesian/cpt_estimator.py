"""Discretización del subset de café y estimación de CPTs con suavizado de Laplace.

`crop_recommendation.csv` no incluye etiqueta de roya ni de rendimiento.
Para alimentar el DAG se generan etiquetas sintéticas determinísticas
(RiesgoRoya) y estocásticas reproducibles (RendimientoEsperado) sobre
el subset `label == 'coffee'`. Las reglas de etiquetado y las
distribuciones condicionales están documentadas abajo y reflejan la
epidemiología publicada de *Hemileia vastatrix* (Avelino et al.).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


STATES = ("bajo", "medio", "alto")
STATE_INDEX = {s: i for i, s in enumerate(STATES)}


def discretize_climate(df: pd.DataFrame) -> pd.DataFrame:
    """Discretiza temperature, humidity y rainfall en {0,1,2}."""
    out = df.copy()
    out["T"] = pd.cut(
        out["temperature"], bins=[-np.inf, 18, 24, np.inf], labels=[0, 1, 2],
    ).astype(int)
    out["H"] = pd.cut(
        out["humidity"], bins=[-np.inf, 60, 80, np.inf], labels=[0, 1, 2],
    ).astype(int)
    out["P"] = pd.cut(
        out["rainfall"], bins=[-np.inf, 1200, 2000, np.inf], labels=[0, 1, 2],
    ).astype(int)
    return out


def label_rust_risk(row: pd.Series) -> int:
    """Etiqueta determinística de RiesgoRoya a partir de T, H, P."""
    h_pts = {0: 0, 1: 1, 2: 2}[row["H"]]
    t_pts = {0: 0, 1: 2, 2: 1}[row["T"]]
    p_pts = {0: 0, 1: 1, 2: 1}[row["P"]]
    score = h_pts + t_pts + p_pts
    if score <= 1:
        return 0
    if score <= 3:
        return 1
    return 2


YIELD_GIVEN_RUST = np.array([
    [0.05, 0.15, 0.80],
    [0.20, 0.60, 0.20],
    [0.70, 0.25, 0.05],
])


def label_yield(rust: int, rng: np.random.Generator) -> int:
    return int(rng.choice(3, p=YIELD_GIVEN_RUST[rust]))


def build_labeled_dataset(coffee_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    df = discretize_climate(coffee_df)
    df["R"] = df.apply(label_rust_risk, axis=1)
    rng = np.random.default_rng(seed)
    df["Y"] = df["R"].map(lambda r: label_yield(r, rng))
    return df[["T", "H", "P", "R", "Y"]]


def estimate_cpt(
    df: pd.DataFrame, target: str, parents: Sequence[str],
    cardinality: Dict[str, int], alpha: float = 1.0,
) -> np.ndarray:
    """Estima P(target | parents) con suavizado Laplace.

    Devuelve un array de forma `(card[target], card[parent_0], ...,
    card[parent_{k-1}])`. Si `parents` está vacío, devuelve la marginal
    como vector de longitud `card[target]`.
    """
    k_target = cardinality[target]
    if not parents:
        counts = np.full(k_target, alpha)
        vc = df[target].value_counts()
        for s, n in vc.items():
            counts[int(s)] += n
        return counts / counts.sum()

    parent_cards = [cardinality[p] for p in parents]
    shape = (k_target, *parent_cards)
    counts = np.full(shape, alpha)
    for _, row in df.iterrows():
        idx = (int(row[target]), *(int(row[p]) for p in parents))
        counts[idx] += 1
    totals = counts.sum(axis=0, keepdims=True)
    return counts / totals

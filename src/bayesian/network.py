"""Representación del DAG y CPTs como factores numpy.

Convención: las CPTs P(X | parents) se almacenan como arrays de forma
`(card[X], card[parents[0]], ..., card[parents[k-1]])`. Las distribuciones
marginales (sin padres) son vectores de longitud `card[X]`.

`Factor` es la unidad de trabajo para Variable Elimination: cada uno guarda
la lista ordenada de variables y un array multidimensional con la misma
forma. Los axes se alinean por nombre de variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class Factor:
    variables: List[str]
    array: np.ndarray

    def __post_init__(self) -> None:
        if len(self.variables) != self.array.ndim:
            raise ValueError(
                f"Factor with vars {self.variables} but array.ndim={self.array.ndim}"
            )


@dataclass
class BayesianNetwork:
    nodes: List[str]
    parents: Dict[str, List[str]]
    cpts: Dict[str, np.ndarray]
    cardinality: Dict[str, int]
    states: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    def factor_for(self, node: str) -> Factor:
        return Factor(
            variables=[node, *self.parents[node]],
            array=self.cpts[node],
        )

    def factors(self) -> List[Factor]:
        return [self.factor_for(n) for n in self.nodes]


def build_coffee_network(
    cpt_T: np.ndarray, cpt_H: np.ndarray, cpt_P: np.ndarray,
    cpt_R_given_THP: np.ndarray, cpt_Y_given_R: np.ndarray,
) -> BayesianNetwork:
    """Construye la red del módulo 2.

    DAG:
        T, H, P → R
        R       → Y
    """
    return BayesianNetwork(
        nodes=["T", "H", "P", "R", "Y"],
        parents={"T": [], "H": [], "P": [], "R": ["T", "H", "P"], "Y": ["R"]},
        cpts={"T": cpt_T, "H": cpt_H, "P": cpt_P,
              "R": cpt_R_given_THP, "Y": cpt_Y_given_R},
        cardinality={"T": 3, "H": 3, "P": 3, "R": 3, "Y": 3},
        states={
            "T": ("baja", "media", "alta"),
            "H": ("baja", "media", "alta"),
            "P": ("baja", "media", "alta"),
            "R": ("bajo", "medio", "alto"),
            "Y": ("bajo", "medio", "alto"),
        },
    )

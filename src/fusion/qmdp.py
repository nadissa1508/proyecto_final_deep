"""Política con creencia (QMDP) sobre el MDP heredado.

El MDP heredado asume que el nivel de roya de la parcela se conoce. Aquí solo se tiene una creencia `b(R)` (de la red bayesiana
con el pronóstico y las fotos). QMDP es la aproximación estándar de POMDP que sigue usando `Q*` de Value Iteration:

    a* = argmax_a  Σ_r  b(r) · Q*( (mes, r), a )

Supone que la incertidumbre se resuelve después de la acción (no valora acciones solo para obtener información), por eso es una
cota optimista del valor de actuar bajo incertidumbre y no sirve para decidir si conviene tomar otra foto. Se documenta en el informe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from src.mdp.environment import ACTION_NAMES, N_RUST, ParcelaProfile, build_environment, state_index
from src.mdp.solver import VISolution, value_iteration


@dataclass
class QmdpResult:
    action: int
    action_name: str
    values: np.ndarray               # Σ_r b(r) Q*((mes, r), a) por acción


def solve_parcel(profile: Optional[ParcelaProfile] = None, gamma: float = 0.95, tol: float = 1e-3) -> VISolution:
    """Value Iteration del MDP heredado para una parcela (o la base si es None); es Q* que usa QMDP."""
    T, R = build_environment(profile)
    return value_iteration(T, R, gamma=gamma, tol=tol)


def qmdp(belief: Sequence[float], month: int, Q: np.ndarray) -> QmdpResult:
    b = np.asarray(belief, dtype=float)
    if b.shape != (N_RUST,) or not np.isclose(b.sum(), 1.0, atol=1e-6) or (b < -1e-12).any():
        raise ValueError("La creencia debe ser una distribución de 3 niveles de roya")
    if not 1 <= month <= 12:
        raise ValueError("month debe estar entre 1 y 12")
    rows = [state_index(month, r) for r in range(N_RUST)]
    values = b @ Q[rows]                       # (acciones,)
    action = int(values.argmax())
    return QmdpResult(action=action, action_name=ACTION_NAMES[action], values=values)

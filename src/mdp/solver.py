"""Value Iteration y policy evaluation desde cero.

Implementa el algoritmo estándar con criterio de parada por norma infinito sobre
el vector de valores. Devuelve V*, π* y el número de iteraciones.

`evaluate_policy` calcula V^π por iteración de punto fijo (Bellman expectation)
para una política dada, útil para comparar la política óptima contra baselines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VISolution:
    V: np.ndarray
    policy: np.ndarray
    Q: np.ndarray
    iterations: int


def value_iteration(
    T: np.ndarray, R: np.ndarray, gamma: float = 0.95,
    tol: float = 1e-3, max_iter: int = 10_000,
) -> VISolution:
    n_states, n_actions = R.shape
    V = np.zeros(n_states)
    for k in range(1, max_iter + 1):
        Q = R + gamma * T.dot(V)
        V_new = Q.max(axis=1)
        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        if delta < tol:
            policy = Q.argmax(axis=1)
            return VISolution(V=V, policy=policy, Q=Q, iterations=k)
    Q = R + gamma * T.dot(V)
    return VISolution(V=V, policy=Q.argmax(axis=1), Q=Q, iterations=max_iter)


def evaluate_policy(
    T: np.ndarray, R: np.ndarray, policy: np.ndarray,
    gamma: float = 0.95, tol: float = 1e-6, max_iter: int = 10_000,
) -> np.ndarray:
    n_states = R.shape[0]
    V = np.zeros(n_states)
    s_idx = np.arange(n_states)
    T_pi = T[s_idx, policy]
    R_pi = R[s_idx, policy]
    for _ in range(max_iter):
        V_new = R_pi + gamma * T_pi.dot(V)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    return V


def constant_policy(action: int, n_states: int) -> np.ndarray:
    return np.full(n_states, action, dtype=int)

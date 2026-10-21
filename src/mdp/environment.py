"""Ambiente MDP para manejo estacional de una parcela cafetalera.

Estado: combinación de mes calendario (1..12) y nivel de roya {bajo, medio, alto}
→ 36 estados totales. Codificación: `state_idx = (mes - 1) * 3 + roya`.

Acciones (5): 0 nada, 1 fertilizar, 2 fungicida cobre (preventivo),
3 fungicida triazol (curativo), 4 podar.

Transiciones T(s, a, s'): el mes siempre avanza (wrap dic→ene). El nivel de
roya transita según una matriz que depende de la temporada (lluviosa
mayo–octubre, seca noviembre–abril), de la acción aplicada y opcionalmente
del perfil físico de la parcela (altitud, sombra, humedad típica).

Reward R(s, a): ingreso bruto por cosecha proporcional al rendimiento esperado
del estado, menos costo de la acción, menos penalización por roya alta
desatendida. Cosecha distribuida en meses octubre–febrero.

Todos los parámetros monetarios son referenciales (USD por hectárea) y están
documentados en este archivo. Pensado para 1 ha.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


N_MONTHS = 12
N_RUST = 3
N_STATES = N_MONTHS * N_RUST
N_ACTIONS = 5

ACTION_NAMES = ("nada", "fertilizar", "cobre", "triazol", "podar")
RUST_NAMES = ("bajo", "medio", "alto")
MONTH_NAMES = (
    "ene", "feb", "mar", "abr", "may", "jun",
    "jul", "ago", "sep", "oct", "nov", "dic",
)

WET_MONTHS = {5, 6, 7, 8, 9, 10}
HARVEST_MONTHS = {10, 11, 12, 1, 2}


@dataclass(frozen=True)
class Parameters:
    coffee_price_per_qq: float = 200.0
    annual_yield_by_rust: tuple = (25.0, 18.0, 8.0)
    harvest_share_per_month: float = 1.0 / len(HARVEST_MONTHS)
    cost_by_action: tuple = (0.0, 50.0, 30.0, 80.0, 100.0)
    high_rust_neglect_penalty: float = 50.0
    discount: float = 0.95


@dataclass(frozen=True)
class ParcelaProfile:
    """Atributos físicos de la parcela que modulan la presión de roya.

    Cuando se pasa a `build_environment`, las matrices de transición se
    ajustan según una función `rust_pressure` ∈ [0.5, 1.5] derivada de
    estos tres factores. Sin parcela, las matrices quedan en su baseline
    (presión = 1.0).
    """
    altitud: float = 1500.0
    sombra: int = 1
    humedad_tipica: float = 70.0


def state_index(month: int, rust: int) -> int:
    return (month - 1) * N_RUST + rust


def decode_state(s: int) -> tuple:
    return (s // N_RUST) + 1, s % N_RUST


def _rust_transition_dry() -> np.ndarray:
    """P(rust' | rust, action) en temporada seca. Shape (rust, action, rust')."""
    T = np.zeros((N_RUST, N_ACTIONS, N_RUST))
    T[0, 0] = [0.85, 0.13, 0.02]
    T[1, 0] = [0.20, 0.65, 0.15]
    T[2, 0] = [0.05, 0.25, 0.70]
    T[0, 1] = [0.85, 0.13, 0.02]
    T[1, 1] = [0.25, 0.65, 0.10]
    T[2, 1] = [0.05, 0.30, 0.65]
    T[0, 2] = [0.95, 0.04, 0.01]
    T[1, 2] = [0.40, 0.50, 0.10]
    T[2, 2] = [0.15, 0.40, 0.45]
    T[0, 3] = [0.97, 0.03, 0.00]
    T[1, 3] = [0.70, 0.25, 0.05]
    T[2, 3] = [0.40, 0.40, 0.20]
    T[0, 4] = [0.90, 0.09, 0.01]
    T[1, 4] = [0.50, 0.45, 0.05]
    T[2, 4] = [0.25, 0.50, 0.25]
    return T


def _rust_transition_wet() -> np.ndarray:
    """P(rust' | rust, action) en temporada lluviosa: presión de roya mayor."""
    T = np.zeros((N_RUST, N_ACTIONS, N_RUST))
    T[0, 0] = [0.55, 0.35, 0.10]
    T[1, 0] = [0.05, 0.55, 0.40]
    T[2, 0] = [0.02, 0.13, 0.85]
    T[0, 1] = [0.55, 0.35, 0.10]
    T[1, 1] = [0.08, 0.57, 0.35]
    T[2, 1] = [0.02, 0.18, 0.80]
    T[0, 2] = [0.88, 0.10, 0.02]
    T[1, 2] = [0.25, 0.55, 0.20]
    T[2, 2] = [0.10, 0.35, 0.55]
    T[0, 3] = [0.92, 0.07, 0.01]
    T[1, 3] = [0.55, 0.35, 0.10]
    T[2, 3] = [0.30, 0.45, 0.25]
    T[0, 4] = [0.80, 0.17, 0.03]
    T[1, 4] = [0.35, 0.50, 0.15]
    T[2, 4] = [0.15, 0.50, 0.35]
    return T


def rust_pressure(profile: ParcelaProfile) -> float:
    """Factor multiplicativo de presión de roya. Producto de tres factores:
    altitud (frío de altura reduce hongo), sombra (humedad atrapada lo
    favorece) y humedad típica de la parcela. Devuelve un valor en [0.5, 1.5].
    """
    if profile.altitud >= 1800:
        p_alt = 0.45
    elif profile.altitud >= 1400:
        p_alt = 0.75
    elif profile.altitud >= 1100:
        p_alt = 1.00
    else:
        p_alt = 1.40

    p_shade = {0: 0.75, 1: 1.00, 2: 1.25}[int(profile.sombra)]

    if profile.humedad_tipica >= 80:
        p_h = 1.25
    elif profile.humedad_tipica >= 65:
        p_h = 1.05
    elif profile.humedad_tipica >= 55:
        p_h = 0.90
    else:
        p_h = 0.75

    return float(np.clip(p_alt * p_shade * p_h, 0.5, 1.5))


def price_per_qq(profile: ParcelaProfile) -> float:
    """Precio por quintal según altitud. El café de altura (specialty SHB/SHG)
    cotiza más alto que el de bajío (commercial). Valores referenciales en USD.
    """
    if profile.altitud >= 1500:
        return 260.0
    if profile.altitud >= 1200:
        return 210.0
    return 150.0


def _apply_pressure(rust_mat: np.ndarray, pressure: float) -> np.ndarray:
    """Modula la matriz `(rust, action, rust')` por presión de roya.

    Para cada fila (estado actual de roya × acción), las probabilidades
    de empeorar (rust' > rust) se multiplican por `pressure` y las de
    mejorar (rust' < rust) por `1/pressure`. Luego se renormaliza la fila
    para mantener una distribución de probabilidad válida. Si presión = 1,
    la matriz queda intacta.
    """
    if abs(pressure - 1.0) < 1e-9:
        return rust_mat
    adj = rust_mat.copy()
    for i in range(N_RUST):
        for a in range(N_ACTIONS):
            row = adj[i, a, :].copy()
            for j in range(N_RUST):
                if j > i:
                    row[j] *= pressure
                elif j < i:
                    row[j] /= pressure
            s = row.sum()
            if s > 0:
                adj[i, a, :] = row / s
    return adj


def _yield_qq_for_month(month: int, rust: int, params: Parameters) -> float:
    if month not in HARVEST_MONTHS:
        return 0.0
    annual = params.annual_yield_by_rust[rust]
    return annual * params.harvest_share_per_month


def build_environment(
    parcela: Optional[ParcelaProfile] = None,
    params: Parameters = Parameters(),
) -> tuple:
    """Construye (T, R) con shape (N_STATES, N_ACTIONS, N_STATES) y (N_STATES, N_ACTIONS).

    Si `parcela` es None, las matrices de transición usan la presión baseline
    (1.0). Si se provee, las matrices se modulan según `rust_pressure(parcela)`.
    """
    pressure = 1.0 if parcela is None else rust_pressure(parcela)
    price = params.coffee_price_per_qq if parcela is None else price_per_qq(parcela)
    rust_dry = _apply_pressure(_rust_transition_dry(), pressure)
    rust_wet = _apply_pressure(_rust_transition_wet(), pressure)

    T = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    R = np.zeros((N_STATES, N_ACTIONS))

    for month in range(1, N_MONTHS + 1):
        next_month = 1 if month == N_MONTHS else month + 1
        rust_mat = rust_wet if month in WET_MONTHS else rust_dry
        for rust in range(N_RUST):
            s = state_index(month, rust)
            yield_qq = _yield_qq_for_month(month, rust, params)
            revenue = yield_qq * price
            for a in range(N_ACTIONS):
                for rust_next in range(N_RUST):
                    s_next = state_index(next_month, rust_next)
                    T[s, a, s_next] = rust_mat[rust, a, rust_next]
                cost = params.cost_by_action[a]
                penalty = (
                    params.high_rust_neglect_penalty
                    if rust == 2 and a in (0, 1) else 0.0
                )
                R[s, a] = revenue - cost - penalty
    return T, R


def validate_transitions(T: np.ndarray, atol: float = 1e-9) -> None:
    sums = T.sum(axis=2)
    if not np.allclose(sums, 1.0, atol=atol):
        bad = np.argwhere(~np.isclose(sums, 1.0, atol=atol))
        raise ValueError(f"T(s,a,·) no suma 1 en celdas: {bad.tolist()[:5]}")

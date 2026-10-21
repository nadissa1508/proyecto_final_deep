"""Endpoint del módulo MDP.

`GET /api/mdp/policy` devuelve la política óptima baseline (sin condicionar
a parcela), precomputada al startup. `POST /api/mdp/policy` re-resuelve Value
Iteration con presión de roya y precio modulados por la parcela enviada.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.state import get_state
from src.mdp.environment import (
    ACTION_NAMES, MONTH_NAMES, N_RUST, RUST_NAMES, ParcelaProfile,
    build_environment, price_per_qq, rust_pressure, state_index,
)
from src.mdp.solver import value_iteration


router = APIRouter(prefix="/api/mdp", tags=["mdp"])


class ParcelaInput(BaseModel):
    altitud: float = Field(..., ge=0, le=4000)
    sombra: int = Field(..., ge=0, le=2)
    humedad_tipica: float = Field(..., ge=0, le=100)


class PolicyResponse(BaseModel):
    meses: List[str]
    niveles_roya: List[str]
    acciones: List[str]
    politica: List[List[str]]
    valores: List[List[float]]
    iteraciones: int
    rust_pressure: Optional[float] = None
    precio_qq: Optional[float] = None


def _format_response(sol, pressure: Optional[float] = None,
                     price: Optional[float] = None) -> PolicyResponse:
    politica: List[List[str]] = []
    valores: List[List[float]] = []
    for m in range(12):
        row_p, row_v = [], []
        for r in range(N_RUST):
            s = state_index(m + 1, r)
            row_p.append(ACTION_NAMES[int(sol.policy[s])])
            row_v.append(round(float(sol.V[s]), 2))
        politica.append(row_p)
        valores.append(row_v)
    return PolicyResponse(
        meses=list(MONTH_NAMES),
        niveles_roya=list(RUST_NAMES),
        acciones=list(ACTION_NAMES),
        politica=politica,
        valores=valores,
        iteraciones=sol.iterations,
        rust_pressure=pressure,
        precio_qq=price,
    )


@router.get("/policy", response_model=PolicyResponse)
def policy_baseline() -> PolicyResponse:
    sol = get_state().mdp_solution
    if sol is None:
        raise HTTPException(503, "MDP no resuelto")
    return _format_response(sol)


@router.post("/policy", response_model=PolicyResponse)
def policy_for_parcela(parcela: ParcelaInput) -> PolicyResponse:
    profile = ParcelaProfile(
        altitud=parcela.altitud,
        sombra=parcela.sombra,
        humedad_tipica=parcela.humedad_tipica,
    )
    T, R = build_environment(profile)
    sol = value_iteration(T, R, gamma=0.95, tol=1e-3)
    return _format_response(
        sol,
        pressure=round(rust_pressure(profile), 3),
        price=price_per_qq(profile),
    )

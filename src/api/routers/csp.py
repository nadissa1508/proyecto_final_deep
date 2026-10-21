"""Endpoint del módulo CSP."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.state import get_state
from src.csp.solver import Parcela, solve


router = APIRouter(prefix="/api/csp", tags=["csp"])


class ParcelaIn(BaseModel):
    parcela_id: Optional[str] = None
    altitud: float = Field(..., ge=0, le=4000)
    ph: float = Field(..., ge=3.0, le=9.0)
    sombra: int = Field(..., ge=0, le=2)


class SolveRequest(BaseModel):
    parcelas: List[ParcelaIn]
    diversify: bool = True
    w_rust: float = 0.5
    w_yield: float = 0.5


class AssignmentOut(BaseModel):
    parcela_id: str
    variedad: str
    score: float


class SolveResponse(BaseModel):
    asignaciones: List[AssignmentOut]
    score_total: float
    nodos_expandidos: int


@router.post("/solve", response_model=SolveResponse)
def solve_endpoint(req: SolveRequest) -> SolveResponse:
    if not req.parcelas:
        raise HTTPException(400, "Se requiere al menos una parcela")
    state = get_state()
    parcelas = [
        Parcela(
            parcela_id=p.parcela_id or f"P{i+1}",
            altitud=p.altitud, ph=p.ph, sombra_disponible=p.sombra,
        )
        for i, p in enumerate(req.parcelas)
    ]
    try:
        sol = solve(parcelas, state.varieties,
                    diversify=req.diversify, w_rust=req.w_rust, w_yield=req.w_yield)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if sol is None:
        raise HTTPException(422, "No se encontró asignación factible")
    return SolveResponse(
        asignaciones=[
            AssignmentOut(parcela_id=a.parcela_id, variedad=a.variety, score=round(a.score, 4))
            for a in sol.assignments
        ],
        score_total=round(sol.total_score, 4),
        nodos_expandidos=sol.nodes_expanded,
    )

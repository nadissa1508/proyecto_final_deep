"""Endpoint del módulo bayesiano."""

from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.state import get_state
from src.bayesian.inference import variable_elimination


router = APIRouter(prefix="/api/bayesian", tags=["bayesian"])


class InferRequest(BaseModel):
    temperatura: float = Field(..., description="°C")
    precipitacion: float = Field(..., description="mm/temporada")
    humedad: float = Field(..., ge=0, le=100, description="%")


class InferResponse(BaseModel):
    riesgo_roya: Dict[str, float]
    rendimiento: Dict[str, float]


def _discretize_T(t: float) -> int:
    return 0 if t < 18 else (1 if t <= 24 else 2)


def _discretize_H(h: float) -> int:
    return 0 if h < 60 else (1 if h <= 80 else 2)


def _discretize_P(p: float) -> int:
    return 0 if p < 1200 else (1 if p <= 2000 else 2)


@router.post("/infer", response_model=InferResponse)
def infer_endpoint(req: InferRequest) -> InferResponse:
    net = get_state().bayes_net
    if net is None:
        raise HTTPException(503, "Red bayesiana no cargada")
    evidence = {
        "T": _discretize_T(req.temperatura),
        "H": _discretize_H(req.humedad),
        "P": _discretize_P(req.precipitacion),
    }
    post_r = variable_elimination(net, ["R"], evidence)
    post_y = variable_elimination(net, ["Y"], evidence)
    return InferResponse(
        riesgo_roya={
            "bajo": float(post_r.array[0]),
            "medio": float(post_r.array[1]),
            "alto": float(post_r.array[2]),
        },
        rendimiento={
            "bajo": float(post_y.array[0]),
            "medio": float(post_y.array[1]),
            "alto": float(post_y.array[2]),
        },
    )

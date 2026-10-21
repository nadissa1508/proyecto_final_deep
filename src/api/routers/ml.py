"""Endpoint del módulo ML (cultivo recomendado)."""

from __future__ import annotations

from typing import Dict

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.state import get_state


router = APIRouter(prefix="/api/ml", tags=["ml"])


class PredictRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float = Field(..., ge=0, le=100)
    ph: float
    rainfall: float


class PredictResponse(BaseModel):
    cultivo_recomendado: str
    probabilidades: Dict[str, float]
    modelo_usado: str = "GaussianNB (Champion desde cero)"


@router.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest) -> PredictResponse:
    state = get_state()
    if state.ml_model is None or state.ml_scaler is None:
        raise HTTPException(503, "Modelo ML no cargado")
    X = np.array([[req.N, req.P, req.K, req.temperature, req.humidity, req.ph, req.rainfall]])
    Xs = state.ml_scaler.transform(X)
    probs = state.ml_model.predict_proba(Xs)[0]
    top_idx = int(np.argmax(probs))
    top_probs = sorted(
        ((state.ml_classes[i], float(probs[i])) for i in range(len(probs))),
        key=lambda kv: kv[1], reverse=True,
    )[:5]
    return PredictResponse(
        cultivo_recomendado=state.ml_classes[top_idx],
        probabilidades={k: round(v, 4) for k, v in top_probs},
    )

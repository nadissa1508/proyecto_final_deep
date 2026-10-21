"""CoffeeMind GT — backend FastAPI.

Carga al startup (lifespan):
- `varieties.csv` para el solver CSP.
- CPTs del subset coffee de `crop_recommendation.csv` para la red bayesiana.
- Champion (Gaussian Naive Bayes desde cero) entrenado sobre el mismo split del notebook 04.
- Política óptima MDP por Value Iteration.

CORS habilitado para el dev server del frontend en `http://localhost:5173`.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import bayesian, csp, mdp, ml
from src.api.state import load_all


DATA_DIR = Path(os.environ.get("COFFEEMIND_DATA_DIR", "data")).resolve()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all(DATA_DIR)
    yield


app = FastAPI(
    title="CoffeeMind GT API",
    version="0.4.0",
    description="Soporte a decisiones para caficultores guatemaltecos. Cuatro módulos de IA.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(csp.router)
app.include_router(bayesian.router)
app.include_router(ml.router)
app.include_router(mdp.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}

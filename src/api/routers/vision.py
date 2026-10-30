"""Endpoint del módulo de visión: una o varias fotos de hojas → clase, probabilidades y Grad-CAM opcional.

`POST /api/vision/predict` (multipart/form-data, campo `files`). Por hoja devuelve las probabilidades finales (ensamble
calibrado), las de cada modelo, el nivel, si los modelos coinciden, si la foto es concluyente (confianza y contraste suficientes; si no, el motivo) y, si se pide, el mapa
Grad-CAM (PNG en base64).
Responde 503 con el motivo si los pesos no están. El Grad-CAM muestra qué región de la hoja usó el modelo, **no** localiza pústulas.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from src.api.state import get_state
from src.api.uploads import read_images

router = APIRouter(prefix="/api/vision", tags=["vision"])


class LeafResult(BaseModel):
    archivo: str
    clase: str
    indice: int = Field(..., ge=0, le=3)
    confianza: float = Field(..., ge=0, le=1)
    probabilidades: Dict[str, float]
    por_modelo: Dict[str, List[float]]
    modelos_coinciden: bool
    mapa_calor_png_base64: Optional[str] = None
    concluyente: bool = True
    motivo: Optional[str] = None


class PredictResponse(BaseModel):
    hojas: List[LeafResult]
    clases: List[str]
    advertencia: str


@router.post("/predict", response_model=PredictResponse)
async def predict(files: List[UploadFile] = File(...), heatmap: bool = Query(False, description="incluir Grad-CAM")):
    classifier = get_state().vision
    if classifier is None:
        raise HTTPException(503, f"Módulo de visión no disponible: {get_state().vision_error}")
    images = await read_images(files)
    predictions = classifier.predict(images, with_heatmap=heatmap)
    leaves = [LeafResult(archivo=f.filename or f"hoja_{i}", clase=p.label_name, indice=p.label, confianza=p.confidence,
                         probabilidades=dict(zip(classifier.classes, p.probs)), por_modelo=p.per_model,
                         modelos_coinciden=p.agree, mapa_calor_png_base64=p.heatmap_png_base64,
                         concluyente=getattr(p, "usable", True), motivo=getattr(p, "reason", None))
              for i, (f, p) in enumerate(zip(files, predictions))]
    return PredictResponse(
        hojas=leaves, clases=list(classifier.classes),
        advertencia="La clasificación describe cada hoja, no la parcela. Las cifras de evaluación son una cota optimista: "
                    "las fotos de campo pueden diferir de las de los conjuntos de datos.")

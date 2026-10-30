"""Interfaz de inferencia del módulo de visión: la única que usa la API y `src/fusion/`.

Lee un **artefacto** (`models/vision/artifact.json`) que describe qué modelos cargar, y devuelve
para cada hoja la probabilidad de cada clase ya calibrada. No contiene ninguna cifra de resultados:
las temperaturas de calibración, la lista de modelos y los pesos los escribe el proceso de
exportación a partir de lo entrenado y evaluado.

Formato del artefacto:

    {
      "classes": ["sana", "roya_leve", "roya_severa", "otra_afeccion"],
      "img_size": 256,
      "models": [
        {"name": "resnet18_dlr_both", "arch": "resnet18", "weights": "resnet18_dlr_both.pt",
         "temperature": 1.37}
      ],
      "confusion": "confusion_hoja.json",       # opcional: P(O | L) medida en test
      "min_confidence": 0.58,                   # opcional: bajo este valor la foto se marca «no concluyente»
      "min_pixel_std": 10.8                     # opcional: bajo este contraste (gris, 0-255) también
    }

`weights` es relativo a `models/vision/` (o absoluto). Con varios modelos se promedian sus
probabilidades **calibradas** (ensamble) y `agree` indica si todos coinciden en la clase.

`min_confidence` y `min_pixel_std` los escribe `python -m src.vision.abstention --write` (ver su docstring): una foto con poca
confianza o sin contraste se devuelve con `usable=False` y un `reason`, y quien decide no debe contarla como observación. No es un
detector general de «esto no es una hoja»: solo descarta lo dudoso.

Si el artefacto o los pesos no existen se lanza `ModelNotAvailable`, y la API debe responder 503
con un mensaje claro en vez de caerse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

from src.common.paths import MODELS_DIR
from src.vision.explain import GradCAM, denormalize, overlay_png_base64, target_layer
from src.vision.models import build_model
from src.vision.transforms import eval_transform

ARTIFACT_PATH = MODELS_DIR / "vision" / "artifact.json"


class ModelNotAvailable(RuntimeError):
    """No hay artefacto o faltan pesos: el módulo de visión no está listo."""


@dataclass
class LeafPrediction:
    label: int
    label_name: str
    confidence: float
    probs: List[float]
    per_model: Dict[str, List[float]]
    agree: bool
    heatmap_png_base64: Optional[str] = field(default=None, repr=False)
    usable: bool = True
    reason: Optional[str] = None


class LeafClassifier:
    def __init__(self, artifact_path: Path = ARTIFACT_PATH, device: Optional[str] = None) -> None:
        if not artifact_path.exists():
            raise ModelNotAvailable(f"No existe el artefacto de visión: {artifact_path}")
        spec = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.classes: List[str] = spec["classes"]
        self.img_size: int = int(spec["img_size"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.names: List[str] = []
        self.archs: List[str] = []
        self.temperatures: List[float] = []
        self.models: List[torch.nn.Module] = []
        for entry in spec["models"]:
            weights = Path(entry["weights"])
            weights = weights if weights.is_absolute() else MODELS_DIR / "vision" / weights
            if not weights.exists():
                raise ModelNotAvailable(f"Faltan los pesos {weights}")
            model = build_model(entry["arch"], "full_dlr" if entry["arch"] != "small_cnn" else "scratch",
                                pretrained=False)
            model.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
            self.models.append(model.to(self.device).eval())
            self.names.append(entry["name"])
            self.archs.append(entry["arch"])
            self.temperatures.append(float(entry.get("temperature", 1.0)))
        if not self.models:
            raise ModelNotAvailable("El artefacto no lista ningún modelo")
        self.min_confidence: Optional[float] = spec.get("min_confidence")
        self.min_pixel_std: Optional[float] = spec.get("min_pixel_std")
        self._transform = eval_transform(self.img_size)

    def _prepare(self, images: Sequence[Image.Image]) -> torch.Tensor:
        batch = [self._transform(ImageOps.exif_transpose(im).convert("RGB")) for im in images]
        return torch.stack(batch).to(self.device)

    @torch.no_grad()
    def _calibrated_probs(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [torch.softmax(m(x).double() / t, dim=1).cpu() for m, t in zip(self.models, self.temperatures)]

    @staticmethod
    def _pixel_std(image: Image.Image) -> float:
        gray = ImageOps.exif_transpose(image).convert("L").resize((64, 64))
        return float(np.asarray(gray, dtype=np.float64).std())

    def _usability(self, image: Image.Image, confidence: float) -> Optional[str]:
        """Motivo por el que la foto no es concluyente, o None si se puede usar."""
        reasons = []
        if self.min_confidence is not None and confidence < self.min_confidence:
            reasons.append(f"confianza baja ({confidence:.0%})")
        if self.min_pixel_std is not None and self._pixel_std(image) < self.min_pixel_std:
            reasons.append("poco contraste")
        return " y ".join(reasons) or None

    def predict(self, images: Sequence[Image.Image], with_heatmap: bool = False) -> List[LeafPrediction]:
        x = self._prepare(images)
        per_model = self._calibrated_probs(x)
        ensemble = torch.stack(per_model).mean(dim=0)
        predictions = []
        for i in range(len(images)):
            label = int(ensemble[i].argmax())
            heatmap = None
            if with_heatmap:
                with GradCAM(self.models[0], target_layer(self.models[0], self.archs[0])) as g:
                    cam, _, _ = g(x[i:i + 1], torch.tensor([label], device=self.device))
                heatmap = overlay_png_base64(denormalize(x[i]), cam[0].cpu().numpy())
            reason = self._usability(images[i], float(ensemble[i, label]))
            predictions.append(LeafPrediction(
                label=label, label_name=self.classes[label], confidence=float(ensemble[i, label]),
                probs=[float(p) for p in ensemble[i]],
                per_model={n: [float(p) for p in probs[i]] for n, probs in zip(self.names, per_model)},
                agree=len({int(probs[i].argmax()) for probs in per_model}) == 1,
                heatmap_png_base64=heatmap, usable=reason is None, reason=reason))
        return predictions

    def predict_one(self, image: Image.Image, with_heatmap: bool = False) -> LeafPrediction:
        return self.predict([image], with_heatmap)[0]

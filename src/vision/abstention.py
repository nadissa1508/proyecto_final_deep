"""Regla para marcar una foto como «no concluyente» en vez de clasificarla a la fuerza.

El clasificador siempre devuelve una de las 4 clases: una foto que no es una hoja (o una hoja borrosa) recibe de todos modos una etiqueta.
Esta regla no detecta «esto no es una hoja» de forma general (no hay fotos de otras cosas con las que medirlo); solo descarta las fotos
donde el ensamble está poco seguro y las imágenes casi uniformes, y mide qué se gana y qué se pierde con eso.

Regla fijada **antes** de ver el test y con datos de validación únicamente:

- `min_confidence`: percentil 10 de la confianza del ensamble (probabilidad calibrada máxima) en la validación conjunta; es decir, se
  descartaría ≈ 10 % de las hojas válidas. El 10 % es un parámetro de diseño; la tabla muestra otros cortes para que se vea el costo.
- `min_pixel_std`: la mitad de la desviación estándar de gris más baja vista en las imágenes de **validación** (corte conservador: solo
  atrapa imágenes prácticamente constantes). Una versión anterior usaba también el test (hallazgo de la auditoría); se corrigió.

Se reporta en test (BRACOL y RoCoLe) la cobertura y la exactitud de lo que se conserva, y qué pasa con imágenes sintéticas que no son hojas
(ruido, color uniforme, tablero, degradado, manchas). Con `--write` se agregan ambos valores a `models/vision/artifact.json` (volver a correr
`python -m src.vision.finalize` los borra: hay que repetir este paso) y se escribe `experiments/vision/abstention.json`.

Uso:
    python -m src.vision.abstention            # solo mide
    python -m src.vision.abstention --write    # además actualiza el artefacto
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.common.paths import EXPERIMENTS_DIR, MODELS_DIR
from src.vision.calibration import calibrate
from src.vision.finalize import collect_logits
from src.vision.splits import SPLITS_PATH
from src.vision.train import LeafDataset, select
from src.vision.transforms import eval_transform

QUANTILE = 0.10
EXTRA_QUANTILES = (0.0, 0.05, 0.10, 0.20, 0.30)
STD_SIZE = 64


def pixel_std(image: Image.Image) -> float:
    """Desviación estándar de la imagen en gris (0–255) reducida a 64×64: casi 0 para una imagen uniforme."""
    return float(np.asarray(image.convert("L").resize((STD_SIZE, STD_SIZE)), dtype=np.float64).std())


def synthetic_non_leaves(seed: int = 0, size: int = 256) -> Dict[str, Image.Image]:
    rng = np.random.default_rng(seed)
    checker = (np.indices((size, size)).sum(axis=0) // 16 % 2 * 255).astype(np.uint8)
    gradient = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    blobs = np.kron(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8), np.ones((size // 8, size // 8, 1), dtype=np.uint8))
    return {
        "ruido": Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)),
        "verde uniforme": Image.new("RGB", (size, size), (40, 160, 60)),
        "gris uniforme": Image.new("RGB", (size, size), (128, 128, 128)),
        "tablero": Image.fromarray(np.stack([checker] * 3, axis=-1)),
        "degradado": Image.fromarray(np.stack([gradient] * 3, axis=-1)),
        "manchas de color": Image.fromarray(blobs),
    }


def selective_table(conf: np.ndarray, correct: np.ndarray, tau: float) -> Dict[str, float]:
    keep = conf >= tau
    return {"n": int(len(conf)), "cobertura": round(float(keep.mean()), 4),
            "exactitud_total": round(float(correct.mean()), 4),
            "exactitud_conservadas": round(float(correct[keep].mean()), 4) if keep.any() else float("nan")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Mide la regla de «foto no concluyente» y, con --write, la agrega al artefacto.")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_path = MODELS_DIR / "vision" / "artifact.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    members = [(m["name"], m["arch"], float(m["temperature"])) for m in artifact["models"]]

    splits = pd.read_csv(SPLITS_PATH)
    frames = {"val": select(splits, ["bracol", "rocole"], "val"),
              "test_bracol": select(splits, ["bracol"], "test"), "test_rocole": select(splits, ["rocole"], "test")}
    labels = {k: v["label"].astype(int).to_numpy() for k, v in frames.items()}
    probs = {k: 0 for k in frames}
    for name, arch, temperature in members:
        logits = collect_logits(name, arch, frames, device)
        for split in frames:
            probs[split] = probs[split] + calibrate(logits[split], temperature).numpy() / len(members)
        print(f"listo: {name}", flush=True)

    conf = {k: p.max(axis=1) for k, p in probs.items()}
    correct = {k: p.argmax(axis=1) == labels[k] for k, p in probs.items()}
    tau = float(np.quantile(conf["val"], QUANTILE))

    stds = []
    for path in LeafDataset(frames["val"], eval_transform(256)).paths:  # solo validación: el test no fija la regla
        with Image.open(path) as im:
            stds.append(pixel_std(im))
    min_std = round(0.5 * float(min(stds)), 3)

    rows = []
    for q in EXTRA_QUANTILES:
        t = float(np.quantile(conf["val"], q))
        for split in frames:
            rows.append({"cuantil_val": q, "corte": round(t, 4), "conjunto": split, **selective_table(conf[split], correct[split], t)})
    table = pd.DataFrame(rows)

    # Imágenes sintéticas: se pasan por el mismo ensamble calibrado
    from src.vision.infer import LeafClassifier

    classifier = LeafClassifier(artifact_path, device=str(device))
    synthetic = synthetic_non_leaves()
    preds = classifier.predict(list(synthetic.values()))
    synth = []
    for (name, image), pred in zip(synthetic.items(), preds):
        std = pixel_std(image)
        reasons = [r for r, bad in (("confianza baja", pred.confidence < tau), ("poco contraste", std < min_std)) if bad]
        synth.append({"imagen": name, "clase": pred.label_name, "confianza": round(pred.confidence, 3), "desv_gris": round(std, 1),
                      "descartada": bool(reasons), "motivo": " + ".join(reasons)})

    report = {"regla": {"min_confidence": round(tau, 4), "min_pixel_std": min_std, "cuantil_validacion": QUANTILE,
                        "desv_gris_minima_en_validacion": round(float(min(stds)), 3)},
              "cobertura_y_exactitud": json.loads(table.round(4).to_json(orient="records")), "sinteticas": synth}
    out = EXPERIMENTS_DIR / "vision" / "abstention.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(f"\nmin_confidence = {tau:.4f} (cuantil {QUANTILE:.0%} de la validación) · min_pixel_std = {min_std}")
    print(table.round(4).to_string(index=False))
    print("\nImágenes sintéticas que no son hojas:")
    print(pd.DataFrame(synth).to_string(index=False))
    if args.write:
        artifact["min_confidence"] = round(tau, 4)
        artifact["min_pixel_std"] = min_std
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"\nArtefacto actualizado: {artifact_path}")


if __name__ == "__main__":
    main()

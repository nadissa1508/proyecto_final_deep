"""Prueba de punta a punta de `infer.py` con el artefacto real.

Pasa por `LeafClassifier` las imágenes de test de BRACOL y de RoCoLe (las mismas copias redimensionadas que
usó la evaluación) y compara el F1 macro obtenido con el que dejó `finalize.py` en
`experiments/vision/final_comparison.csv` para el candidato elegido. Deben coincidir salvo diferencias
numéricas pequeñas (la evaluación usó precisión mixta en GPU; aquí se usa float32).

Uso:
    python -m src.vision.check_infer              # CPU
    python -m src.vision.check_infer --device cuda
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score

from src.common.paths import EXPERIMENTS_DIR
from src.vision.infer import ARTIFACT_PATH, LeafClassifier
from src.vision.preprocess import cache_path
from src.vision.splits import SPLITS_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    classifier = LeafClassifier(ARTIFACT_PATH, device=args.device)
    chosen = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))["candidato"]
    reference = pd.read_csv(EXPERIMENTS_DIR / "vision" / "final_comparison.csv").set_index("candidato").loc[chosen]
    splits = pd.read_csv(SPLITS_PATH)
    print(f"Candidato {chosen}: {len(classifier.models)} modelos en {args.device}", flush=True)
    for dataset in ("bracol", "rocole"):
        test = splits[(splits["dataset"] == dataset) & (splits["split"] == "test")]
        started, predicted = time.time(), []
        paths = [cache_path(d, p) for d, p in zip(test["dataset"], test["path"])]
        for i in range(0, len(paths), args.batch):
            images = [Image.open(p).convert("RGB") for p in paths[i:i + args.batch]]
            predicted += [p.label for p in classifier.predict(images)]
        f1 = f1_score(test["label"].astype(int), predicted, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        seconds = time.time() - started
        print(f"{dataset}: F1 macro con infer.py {f1:.4f} · referencia de finalize.py {reference[f'{dataset}_f1']:.4f} "
              f"· {len(paths)} imágenes en {seconds:.0f} s ({1000 * seconds / len(paths):.0f} ms por imagen)", flush=True)
        assert abs(f1 - reference[f"{dataset}_f1"]) < 0.02, "infer.py no reproduce la evaluación"
    print("OK: infer.py reproduce la evaluación del candidato elegido")


if __name__ == "__main__":
    main()

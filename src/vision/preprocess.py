"""Caché de imágenes redimensionadas.

Las fotos originales miden hasta 4128×2322 px: decodificarlas en cada época vuelve lento
el entrenamiento. Se guarda una copia de 320×320 px (se estira, igual en train y test) en
`data/processed/vision/cache/<dataset>/<nombre>.jpg`. Las rutas salen de `splits.csv`.

Uso:
    python -m src.vision.preprocess
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from src.common.paths import PROCESSED_DIR
from src.vision.splits import SPLITS_PATH

CACHE_DIR = PROCESSED_DIR / "vision" / "cache"
CACHE_SIZE = 320


def cache_path(dataset: str, original: str) -> Path:
    return CACHE_DIR / dataset / (Path(original).stem + ".jpg")


def _resize(args: tuple) -> None:
    dataset, original = args
    target = cache_path(dataset, original)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(original) as im:
        im.draft("RGB", (CACHE_SIZE * 2, CACHE_SIZE * 2))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.resize((CACHE_SIZE, CACHE_SIZE), Image.BICUBIC).save(target, quality=95)


def main() -> None:
    splits = pd.read_csv(SPLITS_PATH)
    jobs = list(zip(splits["dataset"], splits["path"]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_resize, jobs))
    total = sum(1 for _ in CACHE_DIR.rglob("*.jpg"))
    print(f"Caché lista: {total} imágenes de {CACHE_SIZE}×{CACHE_SIZE} en {CACHE_DIR}")


if __name__ == "__main__":
    main()

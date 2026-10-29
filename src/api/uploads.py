"""Lectura y validación de imágenes subidas por el usuario (archivo o cámara del celular).

Se valida el tipo, el tamaño y que el archivo sea de verdad una imagen, y se reduce a `MAX_SIDE` px antes de pasarla a los
modelos para limitar la memoria. El navegador ya reduce y corrige la orientación; aquí se vuelve a hacer por seguridad.
"""

from __future__ import annotations

import io
from typing import List

from fastapi import HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024
MAX_IMAGES = 40
MAX_SIDE = 1024


async def read_images(files: List[UploadFile]) -> List[Image.Image]:
    if not files:
        raise HTTPException(422, "No se envió ninguna imagen")
    if len(files) > MAX_IMAGES:
        raise HTTPException(413, f"Máximo {MAX_IMAGES} imágenes por solicitud")
    images = []
    for upload in files:
        if upload.content_type not in ALLOWED_TYPES:
            hint = " (las fotos HEIC del iPhone deben convertirse a JPEG)" if "heic" in (upload.content_type or "").lower() else ""
            raise HTTPException(415, f"{upload.filename}: tipo {upload.content_type} no permitido{hint}; use JPEG, PNG o WebP")
        data = await upload.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"{upload.filename}: supera {MAX_BYTES // (1024 * 1024)} MB")
        try:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(422, f"{upload.filename}: no es una imagen válida") from exc
        image = image.convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE))
        images.append(image)
    return images

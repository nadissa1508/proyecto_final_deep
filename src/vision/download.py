"""Descarga de RoCoLe y BRACOL desde la API pública de Mendeley Data.

- **RoCoLe** (`c5yvn32dzg`, v2, CC BY 4.0): 1,560 JPG de hojas de robusta y sus
  anotaciones (CSV, JSON, COCO, XLSX). Se omite `RoCoLe-voc.tar.gz` (~700 MB),
  que repite las anotaciones en formato VOC.
- **BRACOL**: el zip publicado en Mendeley (`yy2k5y8mxg`, v1) está truncado en
  origen (su SHA-256 coincide con el publicado, pero no tiene directorio central
  y solo trae 1,402 de 1,747 hojas). Se usa la copia completa que enlazan los
  autores en github.com/esgario/lara2018 (Google Drive, 279 MB, con hojas,
  síntomas y segmentación). Se descarga con `gdown` y se verifica su SHA-256.

Cada archivo se verifica contra el SHA-256 que publica Mendeley. La descarga
es reanudable: los archivos que ya existen con el hash correcto se saltan.

Salida (no versionada): `data/raw/rocole/` y `data/raw/bracol/`.

Uso:
    python -m src.vision.download              # ambos
    python -m src.vision.download --only bracol
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from src.common.paths import RAW_DIR

API = "https://data.mendeley.com/public-api/datasets/{id}"
DATASETS = {
    "rocole": {"id": "c5yvn32dzg", "skip": {"RoCoLe-voc.tar.gz"}},
}

BRACOL_GDRIVE_ID = "15YHebAGrx1Vhv8-naave-R5o3Uo70jsm"
BRACOL_SHA256 = "e6fc32d504ae33e667475dec4c4aef710aac65ad12f16a401c706445c46c9f89"


def download_bracol() -> None:
    import gdown

    out = RAW_DIR / "bracol"
    zpath = out / "gdrive" / "coffee-datasets.zip"
    zpath.parent.mkdir(parents=True, exist_ok=True)
    if not (zpath.exists() and sha256(zpath) == BRACOL_SHA256):
        gdown.download(id=BRACOL_GDRIVE_ID, output=str(zpath), quiet=False)
        if sha256(zpath) != BRACOL_SHA256:
            raise SystemExit("[bracol] SHA-256 no coincide: el archivo de Google Drive cambió")
    target = out / "extracted"
    if not (target / "coffee-datasets" / "leaf" / "dataset.csv").exists():
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(target)
    n = len(list((target / "coffee-datasets" / "leaf" / "images").glob("*.jpg")))
    print(f"[bracol] {n} hojas en {target}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_file(entry: dict, dest: Path, retries: int = 5) -> str:
    details = entry["content_details"]
    if dest.exists() and sha256(dest) == details["sha256_hash"]:
        return "existente"
    for attempt in range(retries):
        try:
            with requests.get(details["download_url"], stream=True, timeout=120) as resp:
                resp.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        fh.write(chunk)
            if sha256(tmp) != details["sha256_hash"]:
                raise IOError("hash SHA-256 no coincide")
            tmp.replace(dest)
            return "descargado"
        except (requests.RequestException, IOError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"{entry['filename']}: {exc}") from exc
            time.sleep(5 * (attempt + 1))
    return "fallido"


def download_dataset(name: str, workers: int = 6) -> None:
    spec = DATASETS[name]
    meta = requests.get(API.format(id=spec["id"]), timeout=60).json()
    out = RAW_DIR / name
    images = out / "images"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mendeley_metadata.json").write_text(
        json.dumps({k: meta[k] for k in ("id", "doi", "name", "version", "description")},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    jobs = []
    for entry in meta["files"]:
        fname = entry["filename"]
        if fname in spec["skip"]:
            continue
        is_img = fname.lower().endswith((".jpg", ".jpeg", ".png"))
        dest = (images if is_img else out) / fname.strip()
        dest.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((entry, dest))

    status = {"descargado": 0, "existente": 0}
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_file, e, d): d for e, d in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                status[fut.result()] += 1
            except RuntimeError as exc:
                errors.append(str(exc))
            if i % 200 == 0 or i == len(jobs):
                print(f"[{name}] {i}/{len(jobs)} {status} errores={len(errors)}", flush=True)

    if errors:
        print(f"[{name}] {len(errors)} archivos con error:")
        for e in errors[:20]:
            print("   ", e)
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", choices=["rocole", "bracol"], default=None)
    args = parser.parse_args()
    if args.only in (None, "rocole"):
        download_dataset("rocole")
    if args.only in (None, "bracol"):
        download_bracol()


if __name__ == "__main__":
    main()

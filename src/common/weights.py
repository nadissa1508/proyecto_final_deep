"""Empaquetado, publicación y descarga de los pesos entrenados.

Los pesos (`models/vision/`, `models/forecast/`) no se versionan: se publican como un zip en un release del repositorio y un clon limpio
los baja con este script. Solo se empaquetan los archivos que los `artifact.json` realmente usan en inferencia (no los de la ablación),
y `models/weights_manifest.json` (sí versionado, pequeño) guarda el SHA-256 de cada uno para comprobar que la descarga está completa.

    python -m src.common.weights pack                      # genera dist/coffeemind_weights.zip y models/weights_manifest.json
    python -m src.common.weights download --url <url-del-zip>
    python -m src.common.weights download --zip dist/coffeemind_weights.zip     # desde un archivo local
    python -m src.common.weights verify                    # compara los pesos presentes con el manifiesto

La URL también puede venir de la variable de entorno `COFFEEMIND_WEIGHTS_URL`. No se escribe ninguna URL a mano en el código.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

MODELS_DIR = Path("models")
MANIFEST_NAME = "weights_manifest.json"
DEFAULT_ZIP = Path("dist") / "coffeemind_weights.zip"
ENV_URL = "COFFEEMIND_WEIGHTS_URL"
MODULES = ("vision", "forecast")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _weight_names(artifact: dict) -> List[str]:
    """Rutas de pesos (relativas a la carpeta del módulo) que declara un artifact.json."""
    names: List[str] = []
    for m in artifact.get("models", []):  # visión
        names.append(m["weights"])
    members = artifact.get("members", {})  # pronóstico: {"base": [...], "ft": [...]}
    for group in members.values():
        for m in group:
            names.append(m["weights"])
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def required_files(models_dir: Path = MODELS_DIR) -> List[str]:
    """Archivos necesarios para inferir, relativos a `models/` y con `/` (p. ej. `vision/artifact.json`)."""
    files: List[str] = []
    for module in MODULES:
        art_path = models_dir / module / "artifact.json"
        if not art_path.exists():
            raise FileNotFoundError(f"Falta {art_path}: genere el artefacto del módulo antes de empaquetar.")
        artifact = json.loads(art_path.read_text(encoding="utf-8"))
        files.append(f"{module}/artifact.json")
        files.extend(f"{module}/{n}" for n in _weight_names(artifact))
    return files


def build_manifest(models_dir: Path = MODELS_DIR) -> Dict[str, dict]:
    manifest: Dict[str, dict] = {}
    for rel in required_files(models_dir):
        path = models_dir / rel
        if not path.exists():
            raise FileNotFoundError(f"El artefacto referencia {path}, que no existe.")
        manifest[rel] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return manifest


def pack(models_dir: Path = MODELS_DIR, out_zip: Path = DEFAULT_ZIP) -> Dict[str, dict]:
    manifest = build_manifest(models_dir)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for rel in manifest:
            zf.write(models_dir / rel, arcname=rel)
    (models_dir / MANIFEST_NAME).write_text(json.dumps({"files": manifest}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def load_manifest(models_dir: Path = MODELS_DIR) -> Dict[str, dict]:
    path = models_dir / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Se genera con `python -m src.common.weights pack` y se versiona.")
    return json.loads(path.read_text(encoding="utf-8"))["files"]


def verify(models_dir: Path = MODELS_DIR, manifest: Optional[Dict[str, dict]] = None) -> List[str]:
    """Lista de problemas (vacía si todo coincide)."""
    manifest = manifest if manifest is not None else load_manifest(models_dir)
    problems: List[str] = []
    for rel, info in manifest.items():
        path = models_dir / rel
        if not path.exists():
            problems.append(f"falta {rel}")
        elif path.stat().st_size != info["bytes"]:
            problems.append(f"tamaño distinto en {rel}")
        elif sha256(path) != info["sha256"]:
            problems.append(f"SHA-256 distinto en {rel}")
    return problems


def extract(zip_path: Path, models_dir: Path = MODELS_DIR, manifest: Optional[Dict[str, dict]] = None) -> List[str]:
    """Extrae solo los archivos del manifiesto (rechaza rutas fuera de `models/`) y devuelve los problemas de verificación."""
    manifest = manifest if manifest is not None else load_manifest(models_dir)
    root = models_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for rel in manifest:
            if rel not in names:
                continue
            target = (models_dir / rel).resolve()
            if root not in target.parents:
                raise ValueError(f"Ruta insegura en el zip: {rel}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(rel) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
    return verify(models_dir, manifest)


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    part.replace(dest)
    return dest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pack", help="empaqueta los pesos usados en inferencia y escribe el manifiesto")
    dl = sub.add_parser("download", help="descarga (o abre) el zip y extrae los pesos")
    dl.add_argument("--url", default=os.environ.get(ENV_URL), help=f"URL del zip (o variable {ENV_URL})")
    dl.add_argument("--zip", type=Path, help="usar un zip local en vez de descargar")
    sub.add_parser("verify", help="comprueba los pesos presentes contra el manifiesto")
    args = parser.parse_args(argv)

    if args.cmd == "pack":
        manifest = pack()
        total = sum(v["bytes"] for v in manifest.values())
        print(f"{len(manifest)} archivos, {total / 1e6:.1f} MB -> {DEFAULT_ZIP} y {MODELS_DIR / MANIFEST_NAME}")
        print("Suba el zip como asset de un release y versione el manifiesto.")
        return 0

    if args.cmd == "verify":
        problems = verify()
        for p in problems:
            print("PROBLEMA:", p)
        print("Pesos completos y coinciden con el manifiesto." if not problems else f"{len(problems)} problema(s).")
        return 1 if problems else 0

    if args.zip:
        zip_path = args.zip
    elif args.url:
        zip_path = download(args.url, DEFAULT_ZIP)
    else:
        print(f"Indique --url, --zip o la variable de entorno {ENV_URL}. La URL sale del release donde se publicó el zip.", file=sys.stderr)
        return 2
    problems = extract(zip_path)
    for p in problems:
        print("PROBLEMA:", p)
    print("Pesos listos en models/." if not problems else f"Descarga incompleta o corrupta: {len(problems)} problema(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

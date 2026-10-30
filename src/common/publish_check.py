"""Revisión previa a publicar el repositorio: archivos que no deberían subirse, secretos y rutas locales.

    python -m src.common.publish_check

Solo lee (usa `git ls-files` y lee archivos de texto); no modifica nada. Revisa lo que ya está versionado y lo que `git add .` agregaría
(archivos sin seguimiento que `.gitignore` no excluye). Sale con código 1 si hay errores.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

BIG_BYTES = 5 * 1024 * 1024
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".jsx", ".js", ".css", ".html", ".txt", ".toml", ".cfg", ".ini", ".ipynb", ".sh"}
REQUIRED_FILES = ["README.md", "LICENSE", "data/README.md", "requirements.txt", "requirements-dl.txt", "backend.requirements.txt",
                  "models/weights_manifest.json", "models/fusion/evidence.json"]
# ruta (prefijo o sufijo) que nunca debe estar en el repositorio público
FORBIDDEN = [
    (re.compile(r"(^|/)NOTAS_INFORME\.md$"), "notas privadas del informe"),
    (re.compile(r"(^|/)docs/auditoria_y_fallos\.md$"), "documento de auditoría privado"),
    (re.compile(r"(^|/)\.env(\.|$)"), "variables de entorno"),
    (re.compile(r"\.(pt|pth|ckpt|safetensors|pkl)$"), "pesos o modelos serializados (excepto los heredados, ver README)"),
    (re.compile(r"\.(zip|tar|tar\.gz|7z)$"), "archivo comprimido"),
    (re.compile(r"(^|/)node_modules/"), "dependencias de npm"),
    (re.compile(r"(^|/)\.venv/"), "entorno virtual"),
    (re.compile(r"^data/(raw/(rocole|bracol|openmeteo|anacafe|power|insivumeh)|processed)/"), "datos descargados o procesados"),
    (re.compile(r"^contenido del curso/"), "material del curso"),
]
ALLOWED_FORBIDDEN = {"models/yield_model.pkl"}  # modelo heredado del proyecto de IA
# archivos privados que el equipo decidió no versionar ni ignorar: sin seguimiento solo avisan; versionados son un error
PRIVATE_REASONS = {"documento de auditoría privado"}
SECRET_PATTERNS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "clave privada"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "clave de AWS"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), "token de GitHub"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "clave de API"),
    (re.compile(r"(?i)\b(password|passwd|contrase(ñ|n)a|api[_-]?key|secret)\s*[:=]\s*['\"][^'\"\s]{6,}['\"]"), "posible credencial en texto"),
]
LOCAL_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+|/home/[a-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/")


def scan_text(text: str) -> List[Tuple[str, str]]:
    """Hallazgos (tipo, fragmento) en un texto: secretos y rutas locales."""
    found: List[Tuple[str, str]] = []
    for pattern, label in SECRET_PATTERNS:
        for m in pattern.finditer(text):
            found.append((label, m.group(0)[:40]))
    for m in LOCAL_PATH.finditer(text):
        found.append(("ruta local", m.group(0)))
    return found


def forbidden_reason(path: str) -> Optional[str]:
    if path in ALLOWED_FORBIDDEN:
        return None
    for pattern, reason in FORBIDDEN:
        if pattern.search(path):
            return reason
    return None


def _git(*args: str) -> List[str]:
    out = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return [line for line in out.split("\n") if line]


def candidate_files() -> Tuple[List[str], List[str]]:
    tracked = _git("ls-files")
    untracked = _git("ls-files", "--others", "--exclude-standard")
    return tracked, untracked


def main() -> int:
    errors: List[str] = []
    warnings: List[str] = []
    tracked, untracked = candidate_files()
    print(f"{len(tracked)} archivos versionados y {len(untracked)} sin seguimiento que `git add .` agregaría.")

    for label, files in (("versionado", tracked), ("sin seguimiento", untracked)):
        for rel in files:
            path = Path(rel)
            if not path.exists():
                continue
            reason = forbidden_reason(rel)
            if reason and label == "sin seguimiento" and reason in PRIVATE_REASONS:
                warnings.append(f"[{label}] {rel}: {reason}; no la agregue con `git add`")
            elif reason:
                errors.append(f"[{label}] {rel}: {reason}")
            size = path.stat().st_size
            if size > BIG_BYTES:
                warnings.append(f"[{label}] {rel}: {size / 1e6:.1f} MB")
            if path.suffix.lower() in TEXT_SUFFIXES and size < 20 * 1024 * 1024:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for kind, fragment in scan_text(text):
                    (errors if kind != "ruta local" else warnings).append(f"[{label}] {rel}: {kind} «{fragment}»")

    for rel in REQUIRED_FILES:
        if not Path(rel).exists():
            errors.append(f"falta {rel}")

    for w in warnings:
        print("AVISO :", w)
    for e in errors:
        print("ERROR :", e)
    print("Sin errores." if not errors else f"{len(errors)} error(es), {len(warnings)} aviso(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

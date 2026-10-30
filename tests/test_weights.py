"""Empaquetado y verificación de pesos con archivos sintéticos (no necesita los pesos reales)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from src.common import weights


def _make_models(root: Path) -> Path:
    models = root / "models"
    (models / "vision").mkdir(parents=True)
    (models / "forecast" / "ft").mkdir(parents=True)
    (models / "vision" / "a.pt").write_bytes(b"vision-a")
    (models / "vision" / "extra_ablacion.pt").write_bytes(b"no se usa")
    (models / "forecast" / "g.pt").write_bytes(b"forecast-g")
    (models / "forecast" / "ft" / "g_ft.pt").write_bytes(b"forecast-ft")
    (models / "vision" / "artifact.json").write_text(json.dumps({"models": [{"weights": "a.pt"}]}), encoding="utf-8")
    (models / "forecast" / "artifact.json").write_text(
        json.dumps({"members": {"base": [{"weights": "g.pt"}], "ft": [{"weights": "ft/g_ft.pt"}]}}), encoding="utf-8")
    return models


def test_only_files_used_in_inference_are_required(tmp_path):
    models = _make_models(tmp_path)
    files = weights.required_files(models)
    assert "vision/a.pt" in files and "forecast/ft/g_ft.pt" in files
    assert "vision/extra_ablacion.pt" not in files


def test_pack_then_extract_into_clean_folder_verifies(tmp_path):
    models = _make_models(tmp_path)
    out = tmp_path / "dist" / "w.zip"
    manifest = weights.pack(models, out)
    assert (models / weights.MANIFEST_NAME).exists()
    assert set(manifest) == set(weights.required_files(models))

    clean = tmp_path / "clone" / "models"
    clean.mkdir(parents=True)
    (clean / weights.MANIFEST_NAME).write_text((models / weights.MANIFEST_NAME).read_text(encoding="utf-8"), encoding="utf-8")
    assert weights.verify(clean)  # antes de extraer faltan todos
    assert weights.extract(out, clean) == []
    assert (clean / "forecast" / "ft" / "g_ft.pt").read_bytes() == b"forecast-ft"


def test_verify_detects_corrupted_file(tmp_path):
    models = _make_models(tmp_path)
    weights.pack(models, tmp_path / "w.zip")
    (models / "vision" / "a.pt").write_bytes(b"vision-X")  # mismo tamaño, distinto contenido
    problems = weights.verify(models)
    assert any("SHA-256" in p and "vision/a.pt" in p for p in problems)


def test_missing_referenced_weight_fails_loudly(tmp_path):
    models = _make_models(tmp_path)
    (models / "vision" / "a.pt").unlink()
    with pytest.raises(FileNotFoundError):
        weights.build_manifest(models)


def test_extract_rejects_paths_outside_models(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../fuera.pt", b"x")
    with pytest.raises(ValueError):
        weights.extract(evil, models, {"../fuera.pt": {"sha256": "0", "bytes": 1}})

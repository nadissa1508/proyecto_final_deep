"""Lectura de `configs/fusion.yaml` (parámetros y supuestos de la integración)."""

from __future__ import annotations

from typing import Optional

import yaml

from src.common.paths import CONFIGS_DIR


def load_config(overrides: Optional[dict] = None) -> dict:
    """Lee el YAML y aplica `overrides` anidados (usado por el análisis de sensibilidad)."""
    with open(CONFIGS_DIR / "fusion.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg

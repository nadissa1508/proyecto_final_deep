"""Rutas del proyecto resueltas desde la raíz del repositorio.

Todas las rutas de datos, modelos y experimentos se construyen aquí para que
scripts, notebooks y la API apunten a los mismos lugares. `COFFEEMIND_DATA_DIR`
permite mover `data/` (lo usa el contenedor Docker).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("COFFEEMIND_DATA_DIR", ROOT / "data")).resolve()
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
CONFIGS_DIR = ROOT / "configs"
EXPERIMENTS_DIR = ROOT / "experiments"

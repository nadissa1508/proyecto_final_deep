"""Carga y validación de varieties.csv para el solver CSP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd


@dataclass(frozen=True)
class Variety:
    name: str
    alt_min: float
    alt_max: float
    ph_min: float
    ph_max: float
    shade_required: int
    rust_resistance: int
    yield_score: int


REQUIRED_COLUMNS = [
    "variety", "alt_min", "alt_max", "ph_min", "ph_max",
    "shade_required", "rust_resistance", "yield_score",
]


def load_varieties(path: Union[str, Path]) -> list[Variety]:
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"varieties.csv missing columns: {sorted(missing)}")

    varieties: list[Variety] = []
    for _, row in df.iterrows():
        if row["alt_min"] > row["alt_max"]:
            raise ValueError(f"{row['variety']}: alt_min > alt_max")
        if row["ph_min"] > row["ph_max"]:
            raise ValueError(f"{row['variety']}: ph_min > ph_max")
        if not 0 <= int(row["shade_required"]) <= 2:
            raise ValueError(f"{row['variety']}: shade_required out of [0,2]")
        varieties.append(Variety(
            name=str(row["variety"]),
            alt_min=float(row["alt_min"]),
            alt_max=float(row["alt_max"]),
            ph_min=float(row["ph_min"]),
            ph_max=float(row["ph_max"]),
            shade_required=int(row["shade_required"]),
            rust_resistance=int(row["rust_resistance"]),
            yield_score=int(row["yield_score"]),
        ))
    return varieties


def satisfies_hard_constraints(
    variety: Variety, altitud: float, ph: float, sombra_disponible: int,
) -> bool:
    return (
        variety.alt_min <= altitud <= variety.alt_max
        and variety.ph_min <= ph <= variety.ph_max
        and variety.shade_required <= sombra_disponible
    )

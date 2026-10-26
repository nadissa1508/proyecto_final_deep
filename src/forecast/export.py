"""Exporta el artefacto de pronóstico (`models/forecast/artifact.json`) que lee `infer.py`.

Qué modelo se usa para cada variable sale de la validación en una estación NO vista
(`experiments/forecast/loso_eval.csv`, ver `loso.py`), no de una decisión a mano: para cada variable se
compara el MAE medio (a 7 y 14 días) del modelo afinado con el del preentrenado y se elige el menor.
Con la familia GRU (empata con el Transformer y es más simple) y todas sus semillas como ensamble.

También se guardan:
- los normalizadores de cada región (media/desv. de ERA5 en train), porque el modelo trabaja en escala normalizada;
- el horizonte mínimo confiable cuando la entrada está atrasada: el menor horizonte cuyo aumento de error al
  rellenar 5 días por persistencia (`latency_staleness.csv`) no supera `STALE_TOLERANCE` en ninguna variable.

Uso:
    python -m src.forecast.export
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

import pandas as pd
import yaml

from src.common.paths import EXPERIMENTS_DIR, MODELS_DIR
from src.forecast.data import INPUT_FEATURES, all_windows, load_config
from src.forecast.download import load_regions
from src.forecast.station_eval import load_run_configs

FAMILY = "forecast_gru"
FT_STRATEGY = "head"
STALE_DAYS = 5
STALE_TOLERANCE = 0.05
JUDGE_HORIZONS = (7, 14)


def choose_sources(loso: pd.DataFrame, targets: List[str]) -> Dict[str, str]:
    """`ft` o `base` por variable: el de menor MAE medio en la estación no vista a los horizontes de juicio."""
    loso = loso[loso["horizonte"].isin(JUDGE_HORIZONS)].copy()
    loso["fam"] = loso["predictor"].str.replace(r"_s\d$", "", regex=True)
    mae = loso.groupby(["variable", "fam"])["mae"].mean()
    return {t: "ft" if mae[(t, f"loso_ft|{FAMILY}")] < mae[(t, f"preentrenado|{FAMILY}")] else "base" for t in targets}


def min_reliable_horizon(stale: pd.DataFrame, horizons: List[int]) -> int:
    table = stale.groupby(["variable", "horizonte", "k_dias"])["mae"].mean().unstack("k_dias")
    growth = table[STALE_DAYS] / table[0] - 1
    worst = growth.groupby("horizonte").max()
    ok = [h for h in horizons if worst[h] <= STALE_TOLERANCE]
    return min(ok) if ok else max(horizons)


def main() -> None:
    fc = load_config()
    targets, regions = fc["targets"], load_regions()
    sources = choose_sources(pd.read_csv(EXPERIMENTS_DIR / "forecast" / "loso_eval.csv"), targets)
    cfgs = [c for c in load_run_configs() if re.fullmatch(rf"{FAMILY}(_s\d)?", c["name"])]
    members = {"base": [], "ft": []}
    for cfg in cfgs:
        root = cfg["name"]
        seed = re.search(r"(_s\d)$", root)
        ft_name = (root[: seed.start()] if seed else root) + f"_ft{FT_STRATEGY}" + (seed.group(1) if seed else "")
        members["base"].append({"name": root, "weights": f"{root}.pt", "config": cfg})
        members["ft"].append({"name": ft_name, "weights": f"ft/{ft_name}.pt", "config": cfg})
    for entry in members["base"] + members["ft"]:
        if not (MODELS_DIR / "forecast" / entry["weights"]).exists():
            raise SystemExit(f"Falta {entry['weights']}")
    norms = {}
    for region in regions:
        n = all_windows(region, fc)["normalizer"]
        norms[region] = {"mean": n.mean.to_dict(), "std": n.std.to_dict()}
    stale = pd.read_csv(EXPERIMENTS_DIR / "forecast" / "latency_staleness.csv")
    artifact = {
        "features": INPUT_FEATURES, "input_days": fc["input_days"], "max_horizon": fc["max_horizon"],
        "targets": targets, "sources": sources, "members": members, "normalizers": norms,
        "region_coords": {r: {k: s[k] for k in ("lat", "lon", "elevation_m")} for r, s in regions.items()},
        "stale_fill_days": STALE_DAYS,
        "min_horizon_with_gap": min_reliable_horizon(stale, fc["horizons"]),
    }
    path = MODELS_DIR / "forecast" / "artifact.json"
    path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print("fuente por variable:", sources, "| horizonte mínimo con retraso:", artifact["min_horizon_with_gap"])
    print("escrito", path)


if __name__ == "__main__":
    main()

"""Re-evalúa todas las corridas de pronóstico guardadas y las resume (media ± desviación por modelo).

Vuelve a calcular el MAE en el test con los pesos de `models/forecast/<nombre>.pt` y la
configuración guardada en cada corrida, sin reentrenar. Sirve para agregar baselines nuevos a
corridas viejas y para que todas se comparen con el mismo código de evaluación.

Baselines (el mejor de los tres es el rival a vencer):
- persistencia: el valor de hoy;
- estacional: climatología (media) del día del año con el periodo de train;
- estacional mediana: lo mismo con la mediana, el predictor óptimo para el error absoluto;
- estacional + anomalía: climatología más la anomalía media de los últimos 30 días.

El rival a vencer es el mejor de los cuatro en cada región, variable y horizonte.

Salida: `experiments/forecast/summary.csv` (media ± desviación por modelo, variable y horizonte)
y `experiments/forecast/wins.csv`.

Uso:
    python -m src.forecast.evaluate
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import yaml

from src.common.paths import EXPERIMENTS_DIR, MODELS_DIR
from src.forecast.data import load_config, split_windows
from src.forecast.download import load_regions
from src.forecast.models import build_model
from src.forecast.train import evaluate

BASELINES = ["mae_persistencia", "mae_estacional", "mae_estacional_mediana", "mae_estacional_anomalia"]


def re_evaluate(run_dir: Path, device: torch.device) -> pd.DataFrame:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    fc = load_config()
    sample = split_windows(next(iter(load_regions())), fc)["train"]["x"]
    model = build_model(cfg, sample.shape[2], len(fc["targets"]), fc["max_horizon"]).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "forecast" / f"{cfg['name']}.pt", map_location=device))
    table = evaluate(model, cfg, device, bool(cfg["amp"]) and device.type == "cuda")
    table.to_csv(run_dir / "test_mae.csv", index=False)
    table.insert(0, "corrida", cfg["name"])
    table.insert(1, "modelo", cfg["model"])
    table.insert(2, "semilla", cfg["seed"])
    return table


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = sorted(d for d in (EXPERIMENTS_DIR / "forecast").iterdir()
                  if (d / "config.yaml").exists() and (d / "results.json").exists() and "smoke" not in d.name)
    full = pd.concat([re_evaluate(d, device) for d in runs], ignore_index=True)
    full["mejor_baseline"] = full[BASELINES].min(axis=1)
    keys = ["modelo", "variable", "horizonte"]
    # primero se promedian las 6 regiones dentro de cada semilla; luego media ± desviación entre semillas
    per_seed = full.groupby(keys + ["semilla"])[["mae_modelo"] + BASELINES + ["mejor_baseline"]].mean().reset_index()
    summary = per_seed.groupby(keys).agg(
        semillas=("semilla", "nunique"), mae_media=("mae_modelo", "mean"), mae_std=("mae_modelo", "std"),
        persistencia=("mae_persistencia", "mean"), estacional=("mae_estacional", "mean"),
        estacional_mediana=("mae_estacional_mediana", "mean"),
        estacional_anomalia=("mae_estacional_anomalia", "mean"), mejor_baseline=("mejor_baseline", "mean")).reset_index()
    summary["mejora_pct"] = (1 - summary["mae_media"] / summary["mejor_baseline"]) * 100
    out_dir = EXPERIMENTS_DIR / "forecast"
    summary.round(4).to_csv(out_dir / "summary.csv", index=False)
    full["gana"] = full["mae_modelo"] < full["mejor_baseline"]
    wins = full.groupby(["modelo", "semilla"])["gana"].agg(["sum", "count"]).reset_index()
    wins.to_csv(out_dir / "wins.csv", index=False)
    pd.set_option("display.width", 220)
    for variable in summary["variable"].unique():
        part = summary[summary["variable"] == variable]
        print(f"\n== {variable}: MAE medio de las 6 regiones (media ± desv. entre semillas)")
        for h, g in part.groupby("horizonte"):
            base = g.iloc[0]
            line = " | ".join(f"{r.modelo} {r.mae_media:.2f}±{r.mae_std:.2f} ({r.mejora_pct:+.0f}%)" for r in g.itertuples())
            print(f"  h={h:<2d} persist {base.persistencia:.2f} · estac {base.estacional:.2f} · estac+anom "
                  f"{base.estacional_anomalia:.2f} · estac-mediana {base.estacional_mediana:.2f} || {line}")
    print("\nVictorias sobre el mejor de los 4 baselines (de 72 por corrida):")
    print(wins.groupby("modelo")["sum"].agg(["mean", "std", "min", "max", "count"]).round(1).to_string())


if __name__ == "__main__":
    main()

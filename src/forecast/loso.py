"""Validación del afinado dejando una estación fuera (leave-one-station-out), para una finca nueva.

El afinado de `finetune.py` usa las mismas 6 estaciones para ajustar y para evaluar (en periodos distintos).
Eso responde "¿mejora en una estación que ya conocemos?", no "¿mejora en una finca sin estación?", que es el
caso real de un productor. Aquí, para cada región `R`:

1. se afina el modelo preentrenado (ERA5) solo con las otras 5 estaciones (periodo de ajuste y validación);
2. se evalúa en `R` en el periodo de test (mismo protocolo que `station_eval.py`).

Predictores comparados en `R` (todos con las mismas filas):
- `clima_era5`, `numerico`: climatología mediana y pronóstico numérico de Open-Meteo.
- `preentrenado`: modelo entrenado con ERA5, sin ver ninguna estación.
- `preentrenado+sesgo_otras`: el anterior más el sesgo medio contra las OTRAS 5 estaciones (lo único que se
  puede estimar en una finca nueva).
- `loso_ft`: modelo afinado con las otras 5 estaciones; `loso_ft+sesgo_otras`: con el sesgo residual de las otras.
- `preentrenado+sesgo_propio`: usa la estación de `R` en su periodo de ajuste; es una **referencia** (cota de lo
  que se ganaría instalando una estación), no un predictor disponible en una finca nueva.

Uso:
    python -m src.forecast.loso
    python -m src.forecast.loso --bases forecast_gru --strategy full
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from src.common.paths import EXPERIMENTS_DIR, RAW_DIR
from src.forecast.anacafe import load_clean
from src.forecast.baselines import LEAD_COLUMN, climatology
from src.forecast.data import all_windows, load_config, load_region, split_bounds
from src.forecast.download import load_regions
from src.forecast.finetune import finetune_one, station_windows
from src.forecast.station_eval import STATION_COL, load_models, load_run_configs
from src.forecast.train import predict, to_physical

OUT = EXPERIMENTS_DIR / "forecast" / "loso_eval.csv"


def region_context(region: str, spec: dict, fc: dict, stations: pd.DataFrame) -> dict:
    """Todo lo que se necesita de una región: ventanas ERA5, objetivos normalizados de estación y series."""
    df, w = load_region(region), all_windows(region, fc)
    x, y, dates = station_windows(region, spec, fc, stations)
    station = stations[stations["station_id"] == spec["estacion_anacafe"]].set_index("date").reindex(df.index)
    prev = pd.read_csv(RAW_DIR / "openmeteo" / f"previous_runs_{region}.csv",
                       parse_dates=["date"]).set_index("date").reindex(df.index)
    return {"df": df, "w": w, "x": x, "y": y, "dates": dates, "prev": prev,
            "truth": {t: station[STATION_COL[t]].to_numpy() for t in fc["targets"]}}


def pooled_sets(ctx: Dict[str, dict], regions: List[str], fc: dict) -> Dict[str, tuple]:
    """Ventanas de ajuste y validación (mismas fechas que `finetune.build_sets`) de un subconjunto de regiones."""
    st = fc["stations"]
    lo, hi, val_hi = pd.Timestamp(st["adjust_start"]), pd.Timestamp(st["adjust_end"]), pd.Timestamp(st["val_end"])
    parts = {"ajuste": ([], []), "val": ([], [])}
    for r in regions:
        c = ctx[r]
        first, last = c["dates"] + pd.Timedelta(days=1), c["dates"] + pd.Timedelta(days=fc["max_horizon"])
        masks = {"ajuste": (first >= lo) & (last <= hi), "val": (first > hi) & (last <= val_hi)}
        for name, mask in masks.items():
            keep = np.asarray(mask) & ~np.isnan(c["y"]).all(axis=(1, 2))
            parts[name][0].append(c["x"][keep])
            parts[name][1].append(c["y"][keep])
    return {n: (np.concatenate(xs), np.concatenate(ys)) for n, (xs, ys) in parts.items()}


def physical(model: torch.nn.Module, c: dict, fc: dict, device: torch.device) -> np.ndarray:
    return to_physical(predict(model.to(device), c["w"]["x"], device, False), c["w"]["normalizer"], fc["targets"])


def slice_for(c: dict, fc: dict, h: int):
    """Verdad de la estación, fechas objetivo y máscara del periodo de ajuste para el horizonte `h`."""
    st = fc["stations"]
    pos = c["w"]["t_index"] + h
    dates = c["df"].index[pos]
    adjust = np.asarray((dates >= pd.Timestamp(st["adjust_start"])) & (dates <= pd.Timestamp(st["adjust_end"])))
    return pos, dates, adjust


def mean_bias(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    ok = mask & ~np.isnan(truth) & ~np.isnan(pred)
    return float((truth[ok] - pred[ok]).mean()) if ok.any() else float("nan")


def pooled_bias(ctx: Dict[str, dict], others: List[str], preds: Dict[str, np.ndarray], fc: dict,
                target: str, j: int, h: int) -> float:
    """Media (entre las otras estaciones) del sesgo medio de un modelo en su periodo de ajuste."""
    biases = []
    for r in others:
        pos, _, adjust = slice_for(ctx[r], fc, h)
        b = mean_bias(ctx[r]["truth"][target][pos], preds[r][:, h - 1, j], adjust)
        if not np.isnan(b):
            biases.append(b)
    return float(np.mean(biases))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bases", nargs="*", default=["forecast_gru", "forecast_transformer"])
    parser.add_argument("--strategy", default="head", choices=["head", "full"])
    parser.add_argument("--cut", default=None, help="corte alterno de `stations_alt` en configs/forecast.yaml")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fc = load_config()
    if args.cut:
        fc["stations"] = fc["stations_alt"][args.cut]
    out_path = OUT.with_name(f"loso_eval_{args.cut}.csv") if args.cut else OUT
    stations, _ = load_clean()
    regions = load_regions()
    ctx = {r: region_context(r, spec, fc, stations) for r, spec in regions.items()}

    cfgs = [c for c in load_run_configs() if any(re.fullmatch(rf"{b}(_s\d)?", c["name"]) for b in args.bases)]
    first = ctx[next(iter(ctx))]["w"]["x"]
    n_feat, n_tgt, horizon = first.shape[2], len(fc["targets"]), fc["max_horizon"]
    raw_models = load_models(cfgs, n_feat, n_tgt, horizon)
    raw = {name: {r: physical(m, ctx[r], fc, device) for r in ctx} for name, m in raw_models.items()}
    bounds, test_lo = split_bounds(fc), pd.Timestamp(fc["stations"]["test_start"])

    rows = []
    for hold in ctx:
        others = [r for r in ctx if r != hold]
        data = pooled_sets(ctx, others, fc)
        tuned = {}
        for cfg in cfgs:
            model = load_models([cfg], n_feat, n_tgt, horizon)[cfg["name"]]
            info = finetune_one(model, args.strategy, data, device, cfg["seed"])
            tuned[cfg["name"]] = {r: physical(model, ctx[r], fc, device) for r in ctx}
            print(f"{hold} | {cfg['name']}: val Huber {info['val_inicial']:.4f} -> {info['val_mejor']:.4f} "
                  f"({info['epocas']} épocas)", flush=True)
        c = ctx[hold]
        for j, target in enumerate(fc["targets"]):
            clim = climatology(c["df"], target, bounds["train"][1], fc["seasonal_window_days"], stat="median")
            for h in fc["horizons"]:
                pos, dates, adjust = slice_for(c, fc, h)
                truth = c["truth"][target][pos]
                cols = {"clima_era5": clim.reindex(np.minimum(dates.dayofyear.to_numpy(), 365)).to_numpy()}
                if h <= 7:
                    cols["numerico"] = c["prev"][LEAD_COLUMN[target].format(h=h)].to_numpy()[pos]

                def add(key, values):
                    cols[key] = np.clip(values, 0, None) if target == "precipitation_sum" else values

                for name in raw:
                    pre, ft = raw[name][hold][:, h - 1, j], tuned[name][hold][:, h - 1, j]
                    add(f"preentrenado|{name}", pre)
                    add(f"preentrenado+sesgo_otras|{name}", pre + pooled_bias(ctx, others, raw[name], fc, target, j, h))
                    add(f"preentrenado+sesgo_propio|{name}", pre + mean_bias(truth, pre, adjust))
                    add(f"loso_ft|{name}", ft)
                    add(f"loso_ft+sesgo_otras|{name}", ft + pooled_bias(ctx, others, tuned[name], fc, target, j, h))
                valid = np.asarray(dates >= test_lo) & ~np.isnan(truth)
                for v in cols.values():
                    valid &= ~np.isnan(v)
                if valid.sum() < 30:
                    continue
                for key, v in cols.items():
                    rows.append({"region": hold, "variable": target, "horizonte": h, "n": int(valid.sum()),
                                 "predictor": key, "mae": float(np.abs(v[valid] - truth[valid]).mean())})
    table = pd.DataFrame(rows)
    table.round(4).to_csv(out_path, index=False)

    table["familia"] = table["predictor"].str.replace(r"_s\d$", "", regex=True)
    summary = table.groupby(["variable", "horizonte", "familia"])["mae"].mean().unstack("familia")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for variable in fc["targets"]:
        print(f"\n== {variable}: MAE en la estación NO vista, media de las regiones (test desde {test_lo.date()})")
        print(summary.loc[variable].round(2).T.to_string())


if __name__ == "__main__":
    main()

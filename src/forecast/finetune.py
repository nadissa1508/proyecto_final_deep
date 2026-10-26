"""Afinado (transfer learning, S9) de los modelos de pronóstico con las estaciones de ANACAFE.

Los modelos se preentrenaron con ERA5 (entrada ERA5, salida ERA5). Aquí se parte de esos pesos y se
reentrenan con la **misma entrada ERA5** pero con la **estación como objetivo**: aprenden a pronosticar lo
que mide la estación, es decir, a corregir el sesgo de ERA5 y su relación con el terreno local.

Estrategias (las del Lab 6):
- `head`: se congela el modelo y solo se entrena la cabeza lineal (feature extraction).
- `full`: se entrena todo con un learning rate bajo.

Datos y protocolo (fechas del día OBJETIVO, de `configs/forecast.yaml`):
- ajuste: ventanas con los 14 días objetivo dentro de 2022-01-01 … 2024-12-31;
- validación (para parar el entrenamiento): objetivo dentro de 2025-01-01 … 2025-06-30;
- test: nunca se usa aquí; lo evalúa `station_eval.evaluate_region` con el mismo protocolo que los
  baselines, para que "afinado" y "preentrenado + corrección de sesgo" se midan sobre las mismas filas.

Los días en que la estación no tiene dato válido se enmascaran en la pérdida. La pérdida es Huber sobre
las variables ya normalizadas (la lluvia en `log1p`), con el normalizador de ERA5 de cada región.

Uso:
    python -m src.forecast.finetune                       # GRU y Transformer, 4 semillas, head y full
    python -m src.forecast.finetune --bases forecast_gru --strategies full
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.common.paths import EXPERIMENTS_DIR, MODELS_DIR
from src.common.seed import set_seed
from src.forecast.anacafe import load_clean
from src.forecast.data import all_windows, load_config, load_region
from src.forecast.download import load_regions
from src.forecast.station_eval import STATION_COL, evaluate_region, load_models, load_run_configs
from src.forecast.train import predict

FT_DIR = MODELS_DIR / "forecast" / "ft"
STRATEGIES = ("head", "full")
LR = {"head": 1e-3, "full": 1e-4}
MAX_EPOCHS, PATIENCE, BATCH = 15, 3, 128


def ft_name(base: str, strategy: str) -> str:
    """`forecast_gru_s1` + `full` -> `forecast_gru_ftfull_s1` (la semilla al final, para agrupar por familia)."""
    seed = re.search(r"(_s\d)$", base)
    root = base[: seed.start()] if seed else base
    return f"{root}_ft{strategy}{seed.group(1) if seed else ''}"


def station_windows(region: str, spec: dict, fc: dict, stations: pd.DataFrame
                    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Entradas ERA5 y objetivos de estación (normalizados, NaN donde falta dato) para una región."""
    df, w = load_region(region), all_windows(region, fc)
    station = stations[stations["station_id"] == spec["estacion_anacafe"]].set_index("date").reindex(df.index)
    t_idx, norm = w["t_index"], w["normalizer"]
    y = np.full((len(t_idx), fc["max_horizon"], len(fc["targets"])), np.nan, dtype=np.float32)
    for j, target in enumerate(fc["targets"]):
        series = station[STATION_COL[target]].to_numpy(dtype=float)
        if target == "precipitation_sum":
            series = np.log1p(np.clip(series, 0, None))
        series = (series - norm.mean[target]) / norm.std[target]
        for h in range(1, fc["max_horizon"] + 1):
            y[:, h - 1, j] = series[t_idx + h]
    return w["x"], y, w["dates"]


def build_sets(fc: dict, stations: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    st = fc["stations"]
    lo, hi, val_hi = pd.Timestamp(st["adjust_start"]), pd.Timestamp(st["adjust_end"]), pd.Timestamp(st["val_end"])
    parts = {"ajuste": ([], []), "val": ([], [])}
    for region, spec in load_regions().items():
        x, y, dates = station_windows(region, spec, fc, stations)
        first, last = dates + pd.Timedelta(days=1), dates + pd.Timedelta(days=fc["max_horizon"])
        masks = {"ajuste": (first >= lo) & (last <= hi),
                 "val": (first > hi) & (last <= val_hi)}
        for name, mask in masks.items():
            keep = np.asarray(mask) & ~np.isnan(y).all(axis=(1, 2))
            parts[name][0].append(x[keep])
            parts[name][1].append(y[keep])
    return {name: (np.concatenate(xs), np.concatenate(ys)) for name, (xs, ys) in parts.items()}


def masked_huber(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(target)
    if not mask.any():
        return pred.sum() * 0.0
    return nn.functional.huber_loss(pred[mask], target[mask], delta=1.0)


def freeze_for(model: nn.Module, strategy: str) -> List[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = strategy == "full"
    if strategy == "head":
        for p in model.head.parameters():
            p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def val_loss(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    pred = torch.from_numpy(predict(model.to(device), x, device, False))
    return float(masked_huber(pred, torch.from_numpy(y)))


def finetune_one(model: nn.Module, strategy: str, data: dict, device: torch.device, seed: int) -> dict:
    set_seed(seed)
    params = freeze_for(model, strategy)
    model.to(device)
    optimizer = torch.optim.AdamW(params, lr=LR[strategy], weight_decay=0.01)
    (x_tr, y_tr), (x_va, y_va) = data["ajuste"], data["val"]
    x_t, y_t = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    start = val_loss(model, x_va, y_va, device)
    best, best_state, stale, history = start, {k: v.clone() for k, v in model.state_dict().items()}, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(x_t))
        losses = []
        for i in range(0, len(order) - BATCH + 1, BATCH):
            idx = order[i:i + BATCH]
            xb, yb = x_t[idx].to(device), y_t[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_huber(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(loss.item())
        current = val_loss(model, x_va, y_va, device)
        history.append({"epoca": epoch, "loss_train": float(np.mean(losses)), "loss_val": current})
        if current < best:
            best, best_state, stale = current, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return {"val_inicial": start, "val_mejor": best, "epocas": len(history), "historia": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bases", nargs="*", default=["forecast_gru", "forecast_transformer"],
                        help="familias de modelos preentrenados (se usan todas sus semillas)")
    parser.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fc = load_config()
    stations, _ = load_clean()
    data = build_sets(fc, stations)
    print(f"ventanas de ajuste {len(data['ajuste'][0])} · validación {len(data['val'][0])}", flush=True)

    cfgs = [c for c in load_run_configs() if any(re.fullmatch(rf"{b}(_s\d)?", c["name"]) for b in args.bases)]
    sample = all_windows(next(iter(load_regions())), fc)
    n_feat, n_tgt, horizon = sample["x"].shape[2], len(fc["targets"]), fc["max_horizon"]
    base_models = load_models(cfgs, n_feat, n_tgt, horizon)
    FT_DIR.mkdir(parents=True, exist_ok=True)
    tuned, log = {}, []
    for cfg in cfgs:
        for strategy in args.strategies:
            model = load_models([cfg], n_feat, n_tgt, horizon)[cfg["name"]]
            info = finetune_one(model, strategy, data, device, cfg["seed"])
            name = ft_name(cfg["name"], strategy)
            torch.save(model.state_dict(), FT_DIR / f"{name}.pt")
            tuned[name] = model.to("cpu").eval()
            log.append({"modelo": name, **{k: v for k, v in info.items() if k != "historia"}})
            print(f"{name}: val Huber {info['val_inicial']:.4f} -> {info['val_mejor']:.4f} "
                  f"({info['epocas']} épocas)", flush=True)
    (EXPERIMENTS_DIR / "forecast" / "finetune_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    models = {**{k: v.to("cpu") for k, v in base_models.items()}, **tuned}
    table = pd.concat([evaluate_region(r, spec, fc, models, stations) for r, spec in load_regions().items()],
                      ignore_index=True)
    table.round(4).to_csv(EXPERIMENTS_DIR / "forecast" / "station_eval_finetune.csv", index=False)
    table["familia"] = table["predictor"].str.replace(r"_s\d(?=\+|$)", "", regex=True)
    summary = table.groupby(["variable", "horizonte", "familia"])["mae"].mean().unstack("familia")
    pd.set_option("display.width", 250)
    for variable in fc["targets"]:
        cols = [c for c in summary.columns if c.startswith(("forecast_", "persistencia_estacion", "clima_era5+sesgo",
                                                            "numerico"))]
        print(f"\n== {variable}: MAE contra la estación, test desde {fc['stations']['test_start']}")
        print(summary.loc[variable, cols].round(2).T.to_string())


if __name__ == "__main__":
    main()

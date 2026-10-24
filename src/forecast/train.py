"""Entrenamiento y evaluación de los modelos de pronóstico.

Uso:
    python -m src.forecast.train --config configs/forecast_lstm.yaml
    python -m src.forecast.train --config configs/forecast_lstm.yaml --smoke

Un único modelo global se entrena con las ventanas de train de las 6 regiones juntas (cada
región normalizada con sus propias estadísticas de train) y se evalúa por región en el test.
Pérdida: Huber sobre las variables normalizadas, igual peso a los 14 días y a las 3 variables.

La evaluación vuelve a unidades físicas (°C, %, mm) y compara, **sobre exactamente las mismas
muestras**, con la persistencia y el estacional ingenuo. Para el horizonte h solo cuentan las
ventanas cuyo día pronosticado (t+h) cae dentro del periodo de test.

Cada corrida crea `experiments/forecast/<fecha>_<nombre>/` con `config.yaml`, `metrics.csv`
(una fila por época) y `results.json`; los pesos van a `models/forecast/<nombre>.pt` (fuera de git).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.common.paths import EXPERIMENTS_DIR, MODELS_DIR
from src.common.seed import set_seed
from src.forecast.baselines import climatology
from src.forecast.data import load_config, load_region, split_bounds, split_windows
from src.forecast.download import load_regions
from src.forecast.models import build_model


def to_physical(scaled: np.ndarray, norm, targets: List[str]) -> np.ndarray:
    """De la escala normalizada a °C, % y mm (la lluvia estaba en log1p)."""
    out = scaled * norm.std[targets].to_numpy() + norm.mean[targets].to_numpy()
    for i, t in enumerate(targets):
        if t == "precipitation_sum":
            out[..., i] = np.clip(np.expm1(out[..., i]), 0, None)
    return out


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device, amp: bool, batch: int = 1024) -> np.ndarray:
    model.eval()
    chunks = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            chunks.append(model(xb).float().cpu().numpy())
    return np.concatenate(chunks)


def evaluate(model: nn.Module, cfg: dict, device: torch.device, amp: bool) -> pd.DataFrame:
    """MAE por región, variable y horizonte del modelo y de los dos baselines, en las mismas muestras."""
    fc = load_config()
    targets, bounds = fc["targets"], split_bounds(fc)
    test_lo, train_end = bounds["test"][0], bounds["train"][1]
    rows = []
    for region in load_regions():
        w = split_windows(region, fc)
        df, t_idx = load_region(region), w["test"]["t_index"]
        pred = to_physical(predict(model, w["test"]["x"], device, amp), w["normalizer"], targets)
        dates = df.index[t_idx]
        for j, target in enumerate(targets):
            values = df[target].to_numpy()
            clim = climatology(df, target, train_end, fc["seasonal_window_days"])
            # anomalía reciente: media de (observado − climatología) en los últimos 30 días hasta t
            clim_all = clim.reindex(np.minimum(df.index.dayofyear.to_numpy(), 365)).to_numpy()
            anomaly = pd.Series(values - clim_all, index=df.index).rolling(30, min_periods=30).mean().to_numpy()
            clim_med = climatology(df, target, train_end, fc["seasonal_window_days"], stat="median")
            for h in fc["horizons"]:
                keep = (dates + pd.Timedelta(days=h)) >= test_lo
                truth = values[t_idx[keep] + h]
                persistence = values[t_idx[keep]]
                doy = np.minimum((dates[keep] + pd.Timedelta(days=h)).dayofyear.to_numpy(), 365)
                seasonal = clim.reindex(doy).to_numpy()
                seasonal_median = clim_med.reindex(doy).to_numpy()
                shifted = seasonal + anomaly[t_idx[keep]]
                if target == "precipitation_sum":
                    shifted = np.clip(shifted, 0, None)
                rows.append({"region": region, "variable": target, "horizonte": h, "n": int(keep.sum()),
                             "mae_modelo": float(np.abs(pred[keep, h - 1, j] - truth).mean()),
                             "mae_persistencia": float(np.abs(persistence - truth).mean()),
                             "mae_estacional": float(np.abs(seasonal - truth).mean()),
                             "mae_estacional_mediana": float(np.abs(seasonal_median - truth).mean()),
                             "mae_estacional_anomalia": float(np.abs(shifted - truth).mean())})
    return pd.DataFrame(rows)


def run(cfg: dict, smoke: bool = False) -> Path:
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["amp"]) and device.type == "cuda"
    fc = load_config()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = EXPERIMENTS_DIR / "forecast" / f"{stamp}_{cfg['name']}{'_smoke' if smoke else ''}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    regions = list(load_regions())[:2] if smoke else list(load_regions())
    parts = [split_windows(r, fc) for r in regions]
    x_train = np.concatenate([p["train"]["x"] for p in parts])
    y_train = np.concatenate([p["train"]["y"] for p in parts])
    x_val = np.concatenate([p["val"]["x"] for p in parts])
    y_val = np.concatenate([p["val"]["y"] for p in parts])
    if smoke:
        x_train, y_train = x_train[:2000], y_train[:2000]

    n_features, n_targets, horizon = x_train.shape[2], y_train.shape[2], y_train.shape[1]
    model = build_model(cfg, n_features, n_targets, horizon).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    x_val_t, y_val_t = torch.from_numpy(x_val), torch.from_numpy(y_val)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    epochs = 2 if smoke else cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=amp)
    criterion = nn.HuberLoss(delta=1.0)
    print(f"[{cfg['name']}] {device} · train {len(x_train)} · val {len(x_val)} · params {n_params}", flush=True)

    history, best, best_epoch, stale = [], float("inf"), 0, 0
    weights = MODELS_DIR / "forecast" / f"{cfg['name']}{'_smoke' if smoke else ''}.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            if cfg.get("input_noise", 0.0) > 0:  # regularización: ruido gaussiano en la entrada normalizada
                xb = xb + cfg["input_noise"] * torch.randn_like(xb)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = criterion(model(xb).float(), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()
        val_pred = torch.from_numpy(predict(model, x_val, device, amp))
        val_loss = float(criterion(val_pred, y_val_t))
        row = {"epoca": epoch, "loss_train": round(float(np.mean(losses)), 4), "loss_val": round(val_loss, 4)}
        history.append(row)
        print(row, flush=True)
        if val_loss < best:
            best, best_epoch, stale = val_loss, epoch, 0
            torch.save(model.state_dict(), weights)
        else:
            stale += 1
            if stale >= cfg["patience"]:
                print(f"Early stopping en la época {epoch}", flush=True)
                break
    pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)

    model.load_state_dict(torch.load(weights, map_location=device))
    table = evaluate(model, cfg, device, amp)
    table.to_csv(run_dir / "test_mae.csv", index=False)
    results = {"nombre": cfg["name"], "configuracion": cfg, "parametros": n_params, "mejor_epoca": best_epoch,
               "loss_val": best, "segundos": round(time.time() - started, 1),
               "loss_train_baja": bool(history[-1]["loss_train"] < history[0]["loss_train"])}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    best_baseline = table[["mae_persistencia", "mae_estacional", "mae_estacional_mediana",
                               "mae_estacional_anomalia"]].min(axis=1)
    wins = float((table["mae_modelo"] < best_baseline).mean())
    print(f"[{cfg['name']}] mejor época {best_epoch} · val {best:.4f} · gana al mejor baseline en "
          f"{wins:.0%} de las combinaciones región×variable×horizonte", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], metavar="CLAVE=VALOR",
                        help="reemplaza claves del YAML (p. ej. lr=3e-4 dropout=0.3) y las agrega al nombre")
    parser.add_argument("--seed", type=int, default=None, help="reemplaza la semilla del YAML y la agrega al nombre")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for item in args.set:
        key, value = item.split("=", 1)
        try:  # YAML lee "3e-4" como texto; se intenta primero como número
            cfg[key] = float(value) if any(c in value for c in ".e") else int(value)
        except ValueError:
            cfg[key] = yaml.safe_load(value)
        cfg["name"] = f"{cfg['name']}_{key}{value}"
    if args.seed is not None:
        cfg["seed"], cfg["name"] = args.seed, f"{cfg['name']}_s{args.seed}"
    run(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()

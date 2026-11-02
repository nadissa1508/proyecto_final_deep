"""¿Acierta el pronóstico el evento «días con condiciones favorables para la roya»? (no necesita datos de incidencia)

La alarma preventiva (`favorability.py`) depende de cuántos de los próximos 7 días serán favorables. Aquí se mide, en el periodo de test
(2021 en adelante, ERA5 como verdad, mismas muestras para todos los predictores), si el pronóstico del GRU (promedio de las 4 semillas base, sin afinar)
acierta ese evento mejor que dos referencias simples:

- **persistencia**: la última semana observada como pronóstico (cuántos de sus 7 días fueron favorables);
- **climatología**: la frecuencia histórica del evento en esa época del año (1980–2017), por región.

Evento de ventana: al menos `min_favorable_days` de los 7 días siguientes son favorables. Se reporta AUC-ROC (calidad del orden, sin elegir umbral) y
precisión, exhaustividad y F1 con la regla tal cual. También el evento de un solo día a 1, 3, 7 y 14 días. Se evalúan los dos perfiles (`literal` y `altura`).

    python -m src.fusion.evaluate_favorability

Escribe `experiments/fusion/favorabilidad_pronostico.csv`. Límites: la verdad es ERA5, no la estación; no usa los modelos afinados ni el retraso de ERA5
en operación; y esto mide el acierto del evento climático, **no** que seguir la alarma evite pérdidas.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from src.common.paths import CONFIGS_DIR, EXPERIMENTS_DIR, MODELS_DIR
from src.forecast.data import load_config, load_region, split_bounds, split_windows
from src.forecast.download import load_regions
from src.forecast.models import build_model
from src.forecast.train import predict, to_physical
from src.fusion.favorability import FavorabilityConfig, favorable_days

SEEDS = ["", "_s1", "_s2", "_s3"]
T_COL, H_COL = "temperature_2m_mean", "relative_humidity_2m_mean"
DAILY_HORIZONS = (1, 3, 7, 14)


def window_counts(fav: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    """Días favorables en los `window` días que empiezan en cada `starts` (índices a la serie diaria `fav`)."""
    csum = np.concatenate([[0], np.cumsum(fav.astype(int))])
    return csum[starts + window] - csum[starts]


def climatology_rate(event: np.ndarray, dates: pd.DatetimeIndex, train_end: pd.Timestamp, smooth: int = 7) -> pd.Series:
    """Frecuencia del evento por día del año con el periodo de train, suavizada ±`smooth` días (circular)."""
    train = pd.Series(event, index=dates).loc[:train_end].dropna()
    by_doy = train.groupby(np.minimum(train.index.dayofyear.to_numpy(), 365)).mean().reindex(range(1, 366)).interpolate(limit_direction="both")
    padded = pd.concat([by_doy.iloc[-smooth:], by_doy, by_doy.iloc[:smooth]])
    smoothed = padded.rolling(2 * smooth + 1, center=True).mean().iloc[smooth:-smooth]
    smoothed.index = range(1, 366)
    return smoothed


def load_ensemble(n_features: int, device: torch.device):
    cfg = yaml.safe_load((CONFIGS_DIR / "forecast_gru.yaml").read_text(encoding="utf-8"))
    models = []
    for suffix in SEEDS:
        model = build_model(cfg, n_features, 3, 14).to(device)
        model.load_state_dict(torch.load(MODELS_DIR / "forecast" / f"forecast_gru{suffix}.pt", map_location=device, weights_only=True))
        models.append(model)
    return models


def metrics(truth: np.ndarray, pred: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    out = {"n": int(len(truth)), "tasa_evento": round(float(truth.mean()), 4)}
    if truth.min() == truth.max():  # sin positivos (o sin negativos): el AUC no existe
        return {**out, "auc": float("nan"), "ap": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    p, r, f, _ = precision_recall_fscore_support(truth, pred, average="binary", zero_division=0)
    return {**out, "auc": round(float(roc_auc_score(truth, score)), 4), "ap": round(float(average_precision_score(truth, score)), 4),
            "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4)}


def evaluate_region(region: str, profile: str, models, device: torch.device, store: Dict) -> None:
    fcfg = load_config()
    fav_cfg = FavorabilityConfig.from_config(profile=profile)
    window, min_days = fav_cfg.window_days, fav_cfg.min_favorable_days
    bounds = split_bounds(fcfg)
    test_lo, train_end = bounds["test"][0], bounds["train"][1]
    w = split_windows(region, fcfg)
    df = load_region(region)
    t_index = w["test"]["t_index"]
    dates = df.index[t_index]
    keep = dates + pd.Timedelta(days=1) >= test_lo           # los 7 días pronosticados caen dentro del test
    t_index, x = t_index[keep], w["test"]["x"][keep]
    targets = list(fcfg["targets"])
    pred = np.mean([to_physical(predict(m, x, device, amp=False), w["normalizer"], targets) for m in models], axis=0)
    pt, ph = pred[..., targets.index(T_COL)], pred[..., targets.index(H_COL)]          # (n, 14)
    obs_t, obs_h = df[T_COL].to_numpy(), df[H_COL].to_numpy()
    fav_obs = favorable_days(obs_t, obs_h, fav_cfg)
    fav_pred = favorable_days(pt, ph, fav_cfg)                                          # (n, 14)

    # Evento de ventana
    truth_count = window_counts(fav_obs, t_index + 1, window)
    truth = truth_count >= min_days
    fc_count = fav_pred[:, :window].sum(axis=1)
    pers_count = window_counts(fav_obs, t_index - window + 1, window)                    # la última semana observada
    event_all = window_counts(fav_obs, np.arange(len(fav_obs) - window), window) >= min_days
    event_series = np.concatenate([event_all, np.full(window, np.nan)])
    clim = climatology_rate(event_series, df.index, train_end)
    clim_score = clim.reindex(np.minimum((df.index[t_index + 1]).dayofyear.to_numpy(), 365)).to_numpy()
    rows = {
        "pronóstico GRU": metrics(truth, fc_count >= min_days, fc_count + 0.0),
        "persistencia (última semana)": metrics(truth, pers_count >= min_days, pers_count + 0.0),
        "climatología": metrics(truth, clim_score >= 0.5, clim_score),
    }
    for name, m in rows.items():
        store.setdefault(("ventana_7d", profile, name), []).append((region, truth, m, {"pred": None}))
    store[("ventana_7d_datos", profile, region)] = (truth, fc_count, pers_count, clim_score)

    # Evento diario a distintos horizontes
    for h in DAILY_HORIZONS:
        truth_d = fav_obs[t_index + h]
        pers_d = fav_obs[t_index]
        clim_d = climatology_rate(fav_obs.astype(float), df.index, train_end).reindex(
            np.minimum(df.index[t_index + h].dayofyear.to_numpy(), 365)).to_numpy()
        daily = {"pronóstico GRU": metrics(truth_d, fav_pred[:, h - 1], fav_pred[:, h - 1].astype(float)),
                 "persistencia (hoy)": metrics(truth_d, pers_d, pers_d.astype(float)),
                 "climatología": metrics(truth_d, clim_d >= 0.5, clim_d)}
        for name, m in daily.items():
            store.setdefault((f"dia_h{h}", profile, name), []).append((region, truth_d, m, None))
        store[(f"dia_h{h}_datos", profile, region)] = (truth_d, fav_pred[:, h - 1], pers_d, clim_d)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regions = list(load_regions())
    n_features = split_windows(regions[0], load_config())["test"]["x"].shape[2]
    models = load_ensemble(n_features, device)
    rows: List[Dict] = []
    for profile in ("altura", "literal"):
        store: Dict = {}
        for region in regions:
            evaluate_region(region, profile, models, device, store)
        for (event, prof, predictor), items in store.items():
            if event.endswith("_datos"):
                continue
            for region, _, m, _ in items:
                rows.append({"perfil": prof, "evento": event, "region": region, "predictor": predictor, **m})
            # Todas las regiones juntas
            data = [store[(f"{event}_datos", prof, r)] for r, *_ in items]
            truth = np.concatenate([d[0] for d in data])
            if event == "ventana_7d":
                min_days = FavorabilityConfig.from_config(profile=prof).min_favorable_days
                fc, pers, clim = (np.concatenate([d[i] for d in data]) for i in (1, 2, 3))
                score = {"pronóstico GRU": fc + 0.0, "persistencia (última semana)": pers + 0.0, "climatología": clim}[predictor]
                pred = {"pronóstico GRU": fc >= min_days, "persistencia (última semana)": pers >= min_days, "climatología": clim >= 0.5}[predictor]
            else:
                fc, pers, clim = (np.concatenate([d[i] for d in data]) for i in (1, 2, 3))
                score = {"pronóstico GRU": fc.astype(float), "persistencia (hoy)": pers.astype(float), "climatología": clim}[predictor]
                pred = {"pronóstico GRU": fc, "persistencia (hoy)": pers, "climatología": clim >= 0.5}[predictor]
            rows.append({"perfil": prof, "evento": event, "region": "todas", "predictor": predictor, **metrics(truth, pred, score)})
    table = pd.DataFrame(rows)
    out = EXPERIMENTS_DIR / "fusion" / "favorabilidad_pronostico.csv"
    table.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    pooled = table[table["region"] == "todas"].drop(columns=["region"])
    print(pooled.to_string(index=False))
    print(f"\nescrito: {out}")


if __name__ == "__main__":
    main()

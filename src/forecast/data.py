"""Datos de pronóstico: carga de ERA5, split temporal, normalización y ventanas.

Todo lo que define el experimento (fechas de corte, largo de la ventana, horizontes,
variables objetivo) sale de `configs/forecast.yaml`.

Tratamiento de las variables:
- `precipitation_sum` se transforma con `log1p` antes de normalizar: tiene muchos ceros y
  una cola larga (de 0 a más de 100 mm), y sin esto la pérdida la dominarían unos pocos días.
- Se agregan `sin/cos` del día del año (el modelo no puede inferir la estación solo de una
  ventana de 90 días).
- La media y la desviación para normalizar se calculan **solo con el periodo de train**.

Una muestra es (entrada, salida): la entrada son los `input_days` días hasta el día t
(inclusive) y la salida los `max_horizon` días siguientes (t+1 … t+14) de las variables objetivo.
El índice t de cada muestra se guarda para poder comparar contra los baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from src.common.paths import CONFIGS_DIR, RAW_DIR

INPUT_FEATURES = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "relative_humidity_2m_mean", "dew_point_2m_mean", "precipitation_sum",
    "precipitation_hours", "et0_fao_evapotranspiration",
]


def load_config() -> dict:
    with open(CONFIGS_DIR / "forecast.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_region(region: str) -> pd.DataFrame:
    path = RAW_DIR / "openmeteo" / f"era5_{region}.csv"
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    if df.index.to_series().diff().dropna().ne(pd.Timedelta(days=1)).any():
        raise ValueError(f"{region}: la serie diaria tiene fechas faltantes")
    if df[INPUT_FEATURES].isna().any().any():
        raise ValueError(f"{region}: la serie tiene valores nulos")
    return df


def split_bounds(cfg: dict) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])
    return {"train": (pd.Timestamp.min, train_end),
            "val": (train_end + pd.Timedelta(days=1), val_end),
            "test": (val_end + pd.Timedelta(days=1), pd.Timestamp.max)}


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """log1p a la lluvia y marcas de estación; las demás columnas quedan en sus unidades."""
    out = df[INPUT_FEATURES].copy()
    out["precipitation_sum"] = np.log1p(out["precipitation_sum"].clip(lower=0))
    doy = out.index.dayofyear.to_numpy()
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


@dataclass
class Normalizer:
    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "Normalizer":
        return cls(frame.mean(), frame.std().replace(0, 1.0))

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        return (frame - self.mean) / self.std


def make_windows(features: np.ndarray, targets: np.ndarray, input_days: int,
                 max_horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ventanas deslizantes con paso 1.

    Devuelve X (n, input_days, F), Y (n, max_horizon, T) y `t_index` (n,): posición del último
    día de la entrada. Y[i, h-1] es el valor del día t_index[i] + h.
    """
    n = len(features) - input_days - max_horizon + 1
    if n <= 0:
        raise ValueError("Serie demasiado corta para el tamaño de ventana")
    t_index = np.arange(input_days - 1, input_days - 1 + n)
    x = np.stack([features[t - input_days + 1: t + 1] for t in t_index])
    y = np.stack([targets[t + 1: t + 1 + max_horizon] for t in t_index])
    return x.astype(np.float32), y.astype(np.float32), t_index


def split_windows(region: str, cfg: dict | None = None) -> Dict[str, object]:
    """Ventanas de train/val/test de una región, con normalización ajustada solo en train.

    Una ventana pertenece al conjunto en el que cae el **último día de su salida**, y en validación además el **primero**: así la
    salida de validación cae entera en validación y la de train nunca toca val/test. En test la evaluación filtra cada horizonte por la
    fecha del día objetivo. La entrada
    de val/test sí puede usar días anteriores (la historia ya es conocida al pronosticar).
    """
    cfg = cfg or load_config()
    frame = transform(load_region(region))
    bounds = split_bounds(cfg)
    norm = Normalizer.fit(frame.loc[: bounds["train"][1]])
    scaled = norm.apply(frame)
    target_cols = [frame.columns.get_loc(t) for t in cfg["targets"]]
    x, y, t_index = make_windows(scaled.to_numpy(), scaled.to_numpy()[:, target_cols],
                                 cfg["input_days"], cfg["max_horizon"])
    first_out_day = frame.index[t_index + 1]
    last_out_day = frame.index[t_index + cfg["max_horizon"]]
    out: Dict[str, object] = {}
    for name, (lo, hi) in bounds.items():
        mask = (last_out_day >= lo) & (last_out_day <= hi)
        if name == "val":
            # Toda la salida de validación debe caer en validación (antes 13 ventanas por región tocaban días de train; hallazgo
            # de la auditoría). Train ya cumple por construcción y en test la evaluación filtra cada horizonte por la fecha del día objetivo.
            mask &= first_out_day >= lo
        out[name] = {"x": x[mask], "y": y[mask], "t_index": t_index[mask],
                     "dates": frame.index[t_index[mask]]}
    out["normalizer"] = norm
    out["columns"] = list(frame.columns)
    out["targets"] = list(cfg["targets"])
    return out

def all_windows(region: str, cfg: dict | None = None) -> Dict[str, object]:
    """Todas las ventanas de una región (sin partir por conjunto), con la normalización ajustada en train.

    Sirve para evaluar contra las estaciones: ahí la fecha del día objetivo decide a qué periodo
    pertenece cada ventana, no las fechas de corte del entrenamiento con ERA5.
    """
    cfg = cfg or load_config()
    frame = transform(load_region(region))
    norm = Normalizer.fit(frame.loc[: split_bounds(cfg)["train"][1]])
    scaled = norm.apply(frame)
    target_cols = [frame.columns.get_loc(t) for t in cfg["targets"]]
    x, y, t_index = make_windows(scaled.to_numpy(), scaled.to_numpy()[:, target_cols],
                                 cfg["input_days"], cfg["max_horizon"])
    return {"x": x, "y": y, "t_index": t_index, "dates": frame.index[t_index], "normalizer": norm,
            "targets": list(cfg["targets"])}

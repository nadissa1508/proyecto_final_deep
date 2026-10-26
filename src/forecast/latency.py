"""Retraso de ERA5 en operación: cuánto cuesta y con qué se puede llenar.

ERA5 se publica con unos 5 días (en nuestra descarga, 8) de retraso, así que al pronosticar "hoy" los últimos `k`
días de la ventana de entrada no existen en ERA5. Dos mediciones:

1. **Costo de la ventana incompleta** (`staleness`): se repiten los valores del último día conocido en los últimos
   `k` días de la entrada (relleno por persistencia) y se mide cuánto crece el error contra ERA5 en el test
   (k = 0 es la evaluación normal). Es el peor caso razonable: no usa ningún dato nuevo.
2. **Qué tan distintos son los datos recientes de la API de pronóstico** (`mismatch`): la API `forecast` de
   Open-Meteo con `past_days` sí trae los días recientes, pero de otro producto que ERA5. Se compara, en los
   días donde ambos existen, variable por variable, para saber si se puede usar como relleno.

Salida: `experiments/forecast/latency_staleness.csv` y `latency_mismatch.csv`.

Uso:
    python -m src.forecast.latency
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch

from src.common.paths import EXPERIMENTS_DIR
from src.forecast.data import INPUT_FEATURES, load_config, load_region, split_windows
from src.forecast.download import _get, _loc_params, load_regions
from src.forecast.station_eval import load_models, load_run_configs
from src.forecast.train import predict, to_physical

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GAPS = (0, 2, 5, 8)
PAST_DAYS = 45


def persistence_fill(x: np.ndarray, k: int) -> np.ndarray:
    """Repite el último día conocido (posición -k-1) en los últimos k días de las variables meteorológicas.

    Las marcas de calendario (`doy_sin/cos`, columnas siguientes) se dejan: el calendario sí se conoce.
    """
    if k == 0:
        return x
    out = x.copy()
    n = len(INPUT_FEATURES)
    out[:, -k:, :n] = x[:, [-k - 1], :n]
    return out


def staleness(base: str = "forecast_gru") -> pd.DataFrame:
    fc = load_config()
    cfgs = [c for c in load_run_configs() if re.fullmatch(rf"{base}(_s\d)?", c["name"])]
    regions = list(load_regions())
    first = split_windows(regions[0], fc)
    models = load_models(cfgs, first["test"]["x"].shape[2], len(fc["targets"]), fc["max_horizon"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for region in regions:
        w = split_windows(region, fc)
        test, norm = w["test"], w["normalizer"]
        truth = to_physical(test["y"], norm, fc["targets"])
        for k in GAPS:
            x = persistence_fill(test["x"], k)
            for name, model in models.items():
                pred = to_physical(predict(model.to(device), x, device, False), norm, fc["targets"])
                for j, target in enumerate(fc["targets"]):
                    for h in fc["horizons"]:
                        rows.append({"region": region, "modelo": name, "k_dias": k, "variable": target, "horizonte": h,
                                     "mae": float(np.abs(pred[:, h - 1, j] - truth[:, h - 1, j]).mean())})
    return pd.DataFrame(rows)


def mismatch() -> pd.DataFrame:
    """Compara la API `forecast` (días pasados) con ERA5 en los días donde ambos existen."""
    rows = []
    for region, spec in load_regions().items():
        params = {**_loc_params(spec), "daily": ",".join(INPUT_FEATURES), "past_days": PAST_DAYS, "forecast_days": 1,
                  "timezone": "America/Guatemala"}
        payload = _get(FORECAST_URL, params, weight=1.0)["daily"]
        api = pd.DataFrame(payload).assign(date=lambda d: pd.to_datetime(d["time"])).set_index("date").drop(columns="time")
        era5 = load_region(region)
        common = api.index.intersection(era5.index)
        for var in INPUT_FEATURES:
            a, e = api.loc[common, var].astype(float), era5.loc[common, var].astype(float)
            ok = a.notna() & e.notna()
            rows.append({"region": region, "variable": var, "dias": int(ok.sum()),
                         "sesgo": float((a[ok] - e[ok]).mean()), "mae": float((a[ok] - e[ok]).abs().mean()),
                         "desv_era5": float(e[ok].std()),
                         "correlacion": float(np.corrcoef(a[ok], e[ok])[0, 1]) if ok.sum() > 2 else float("nan")})
    return pd.DataFrame(rows)


def main() -> None:
    out = EXPERIMENTS_DIR / "forecast"
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    stale = staleness()
    stale.round(4).to_csv(out / "latency_staleness.csv", index=False)
    table = stale.groupby(["variable", "horizonte", "k_dias"])["mae"].mean().unstack("k_dias")
    base = table[0]
    print("== MAE contra ERA5 (test) al repetir el último día conocido; media de regiones y semillas")
    print(table.round(3).to_string())
    print("\n== Aumento relativo del error frente a k = 0")
    print((table.div(base, axis=0) - 1).round(3).to_string())
    diff = mismatch()
    diff.round(4).to_csv(out / "latency_mismatch.csv", index=False)
    print("\n== API forecast (past_days) vs ERA5, mismos días; media de regiones")
    print(diff.groupby("variable")[["dias", "sesgo", "mae", "desv_era5", "correlacion"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()

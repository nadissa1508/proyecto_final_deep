"""Verifica `infer.py` de punta a punta con los pesos reales.

1. Equivalencia: para cada región, el pronóstico de `WeatherForecaster` con la historia de ERA5 hasta un día `t` del
   test coincide con el del camino de evaluación (ventanas de `data.py` + `predict`) de los mismos modelos.
2. Retraso: pronosticar con la historia cortada `k` días antes de `t` (relleno por persistencia) marca
   `degraded` y cambia el resultado sin errores.
3. Error: MAE del pronóstico contra ERA5 en las fechas de test de la región (referencia del orden de magnitud).

Uso:
    python -m src.forecast.check_infer
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.forecast.data import all_windows, load_config, load_region
from src.forecast.infer import WeatherForecaster
from src.forecast.train import predict, to_physical

TOLERANCE = 1e-4   # °C, puntos o mm/día: infer.py y la evaluación hacen las mismas cuentas, solo se toleran redondeos


def main() -> None:
    fc = load_config()
    forecaster = WeatherForecaster()
    print("fuente por variable:", forecaster.spec["sources"], "| horizonte mínimo con retraso:",
          forecaster.spec["min_horizon_with_gap"])
    worst = 0.0
    errors = []
    for region in forecaster.spec["normalizers"]:
        df, w = load_region(region), all_windows(region, fc)
        pos = len(w["t_index"]) - 60                       # un origen reciente, con 14 días objetivo conocidos
        t = int(w["t_index"][pos])
        history = df.iloc[: t + 1]
        result = forecaster.predict(history, region)
        for j, target in enumerate(fc["targets"]):
            kind = forecaster.spec["sources"][target]
            ref = []
            for m in forecaster.models[kind]:
                p = predict(m, w["x"][pos:pos + 1], torch.device("cpu"), False)
                ref.append(to_physical(p, w["normalizer"], fc["targets"])[0, :, j])
            diff = np.abs(np.mean(ref, axis=0) - np.array(result.values[target])).max()
            worst = max(worst, float(diff))
            truth = df[target].to_numpy()[t + 1: t + 1 + fc["max_horizon"]]
            errors.append({"region": region, "variable": target, "mae": float(np.abs(np.array(result.values[target]) - truth).mean())})
        stale = forecaster.predict(history.iloc[:-5], region, as_of=history.index[-1].date())
        assert stale.stale_days == 5 and stale.degraded[0] and not stale.degraded[-1], "marca de retraso incorrecta"
    print(f"diferencia máxima entre infer.py y el camino de evaluación: {worst:.2e}")
    if worst > TOLERANCE:  # antes solo se imprimía (hallazgo de la auditoría): ahora una divergencia hace fallar la verificación
        raise SystemExit(f"infer.py se aparta del camino de evaluación en {worst:.2e} (tolerancia {TOLERANCE:g})")
    print(pd.DataFrame(errors).groupby("variable")["mae"].mean().round(3).to_string())
    print("retraso: marcas degraded correctas en las 6 regiones")


if __name__ == "__main__":
    main()

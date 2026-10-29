"""Diagnóstico: ¿cuánto dice el clima de las dos semanas sobre el nivel de roya del mes en el mundo simulado?

Si el nivel de roya depende sobre todo del mes anterior (persistencia del MDP), el clima de la ventana casi no aporta información
sobre el nivel actual, y ninguna evidencia climática (ni siquiera un pronóstico perfecto) puede ayudar a elegir la acción. Aquí se
mide eso directamente, sin pronosticador: se simulan trayectorias bajo la política "sin información" y se compara, por cada fuerza
del clima `gamma`, el nivel de roya medio con clima desfavorable (z ≤ -2) y favorable (z ≥ 2), y la correlación entre z y el nivel.

Uso:
    python -m src.fusion.diagnose_climate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.paths import EXPERIMENTS_DIR
from src.forecast.data import load_config as load_forecast_config
from src.forecast.data import load_region
from src.forecast.download import load_regions
from src.fusion.network_ext import ExtendedModel
from src.fusion.qmdp import solve_parcel
from src.fusion.simulate import (HORIZON, INITIAL, N_STATE, World, _categorical, observed_cuts, tercile_index, window_stats,
                                 world_matrices)
from src.mdp.environment import ParcelaProfile, rust_pressure

YEARS = (2018, 2019, 2020, 2021)
GAMMAS = (0.0, 0.15, 0.3, 0.6)


def main() -> None:
    from src.fusion.evidence import HUMIDITY
    fc = load_forecast_config()
    train_end = pd.Timestamp(fc["split"]["train_end"])
    model = ExtendedModel.build()
    neutral = model.climate_posterior(None)
    rng = np.random.default_rng(0)
    rows = []
    for gamma in GAMMAS:
        z_all, r_all = [], []
        for region, spec in load_regions().items():
            df = load_region(region)
            cuts = observed_cuts(df, train_end)
            profile = ParcelaProfile(altitud=float(spec["elevation_m"]), sombra=1, humedad_tipica=float(df[HUMIDITY].loc[:train_end].mean()))
            Q = solve_parcel(profile).Q
            trans = world_matrices(World(gamma=gamma), model.cfg, rust_pressure(profile))
            for year in YEARS:
                z = []
                for m in range(1, 13):
                    obs = window_stats(df, pd.Timestamp(year=year, month=m, day=1), HORIZON)
                    z.append(sum(tercile_index(obs[k], cuts[m][k]) for k in "THP") - 3)
                n = 1500
                r = _categorical(np.tile(np.cumsum(INITIAL), (n, 1)), rng.random(n))
                a_prev = np.zeros(n, dtype=int)
                for m in range(1, 13):
                    if m > 1:
                        r = _categorical(trans[(m - 1, z[m - 1])][r, a_prev], rng.random(n))
                    rows_m = [(m - 1) * N_STATE + k for k in range(N_STATE)]
                    a_prev = np.full(n, int((neutral @ Q[rows_m]).argmax()))        # política "sin información"
                    z_all.append(np.full(n, z[m - 1]))
                    r_all.append(r.copy())
        z, r = np.concatenate(z_all), np.concatenate(r_all)
        rows.append({"gamma": gamma, "corr(z, nivel)": float(np.corrcoef(z, r)[0, 1]),
                     "nivel medio z<=-2": float(r[z <= -2].mean()), "nivel medio z>=2": float(r[z >= 2].mean()),
                     "P(alto) z<=-2": float((r[z <= -2] == 2).mean()), "P(alto) z>=2": float((r[z >= 2] == 2).mean())})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(EXPERIMENTS_DIR / "fusion" / "diagnostico_clima.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

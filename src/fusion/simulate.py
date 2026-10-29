"""Valor de la información: cuánto cambia el ingreso de una temporada simulada si el sistema usa el pronóstico y las fotos.

Mundo simulado (SUPUESTOS, declarados porque el resultado depende de ellos):
- El clima es el real: ERA5 de cada región en los años de simulación (por defecto 2018–2021, que ni el preentrenamiento ni el
  afinado con estaciones usaron).
- La roya sigue las transiciones del MDP heredado, de la parcela (altitud de la estación, humedad típica de la región), pero la
  presión de cada mes se multiplica por `exp(gamma · z)`, con z = índices T+H+P − 3 del clima **observado** en las primeras dos
  semanas del mes respecto de lo habitual. `gamma = 0` deja el MDP heredado sin clima real: ahí el clima no debe aportar nada
  (control de cordura). El nivel de roya del mes m se forma con el clima de esas dos semanas, que es justo la ventana pronosticada.
- El nivel inicial se sortea de `INITIAL` (supuesto). Cada hoja fotografiada sale de P(L | R) y la CNN la clasifica con la matriz
  de confusión medida (`world_confusion`, que puede ser distinta de la que cree la red).
- Se simulan 12 meses; el ingreso es la suma de R(s, a) del MDP en los estados verdaderos.

Escenarios (todos eligen por QMDP con el mismo Q*):
  A  techo           nivel de roya conocido
  B0 sin información creencia = clima neutro (equivale al sistema heredado, cuya CPT de R es uniforme en 23 de 27 casos)
  B  clima observado las dos semanas anteriores, con la red nueva (lo que un productor sabría hoy sin pronóstico)
  C  pronóstico      ventana de 14 días pronosticada por el modelo real
  C0 pronóstico perfecto  la ventana REAL de esas dos semanas como evidencia (cota superior de lo que puede aportar el clima)
  D  pronóstico + CNN  C más `n_leaves` hojas fotografiadas
  D0 solo CNN        solo las hojas, sin clima (cuánto aportan las fotos por sí solas)

Las mismas variables aleatorias se reutilizan en todos los escenarios (números aleatorios comunes), así que las diferencias
entre escenarios son pareadas. La incertidumbre se mide entre región-años, no entre réplicas (las réplicas de un mismo clima
no son independientes).

Uso:
    python -m src.fusion.simulate --reps 300
    python -m src.fusion.simulate --reps 200 --sensitivity
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.common.paths import EXPERIMENTS_DIR
from src.forecast.data import load_config as load_forecast_config
from src.forecast.data import load_region
from src.forecast.download import load_regions
from src.forecast.infer import WeatherForecaster
from src.fusion.config import load_config
from src.fusion.contracts import ClimateEvidence, ClimateWindow
from src.fusion.evidence import HUMIDITY, RAIN, TEMPERATURE
from src.fusion.network_ext import N_STATE, ExtendedModel, leaf_given_rust, load_confusion_counts, obs_given_leaf
from src.fusion.pipeline import FusionEngine
from src.mdp.environment import (WET_MONTHS, ParcelaProfile, _apply_pressure, _rust_transition_dry, _rust_transition_wet,
                                 build_environment, rust_pressure, state_index)

SCENARIOS = ("A_techo", "B0_sin_informacion", "B_clima_observado", "C_pronostico", "C0_pronostico_perfecto",
             "D_pronostico_y_CNN", "D0_solo_CNN")
INITIAL = (0.6, 0.3, 0.1)                 # SUPUESTO: reparto del nivel de roya en enero
HORIZON = 14
OUT = EXPERIMENTS_DIR / "fusion"


@dataclass
class World:
    gamma: float = 0.15
    n_leaves: int = 10
    confusion: str = "pooled"             # matriz con la que la CNN se equivoca en el mundo simulado


@dataclass
class Context:
    region: str
    year: int
    Q: np.ndarray
    R: np.ndarray
    pressure: float
    z: List[int]                          # z observado por mes (1..12 → índice 0..11)
    prior_B: List[np.ndarray]
    prior_C: List[np.ndarray]
    prior_perfect: List[np.ndarray]


def observed_cuts(df: pd.DataFrame, train_end: pd.Timestamp, h: int = HORIZON) -> Dict[int, Dict[str, tuple]]:
    """Terciles (q33, q67) de la ventana observada de h días, por mes del primer día, con ERA5 hasta `train_end`."""
    d = df.loc[:train_end]
    stats = pd.DataFrame({"T": d[TEMPERATURE].rolling(h).mean(), "H": d[HUMIDITY].rolling(h).mean(),
                          "P": d[RAIN].rolling(h).sum()}).shift(-(h - 1)).dropna()
    return {m: {k: (float(g[k].quantile(1 / 3)), float(g[k].quantile(2 / 3))) for k in "THP"}
            for m, g in stats.groupby(stats.index.month)}


def window_stats(df: pd.DataFrame, start: pd.Timestamp, h: int = HORIZON) -> Dict[str, float]:
    w = df.loc[start: start + pd.Timedelta(days=h - 1)]
    return {"T": float(w[TEMPERATURE].mean()), "H": float(w[HUMIDITY].mean()), "P": float(w[RAIN].sum())}


def tercile_index(value: float, cuts: tuple) -> int:
    return 0 if value < cuts[0] else (1 if value < cuts[1] else 2)


def hard_evidence(idx: Dict[str, int], region: str, month: int) -> ClimateEvidence:
    one = lambda i: tuple(1.0 if j == i else 0.0 for j in range(3))
    window = ClimateWindow(region=region, month=month, horizon_days=HORIZON, mean_temperature=0.0, mean_humidity=0.0,
                           total_rain=0.0)
    return ClimateEvidence(window=window, p_T=one(idx["T"]), p_H=one(idx["H"]), p_P=one(idx["P"]))


def build_contexts(engine: FusionEngine, forecaster: WeatherForecaster, years: Sequence[int], stale_days: int,
                   regions: Optional[Sequence[str]] = None) -> List[Context]:
    fc = load_forecast_config()
    train_end = pd.Timestamp(fc["split"]["train_end"])
    contexts = []
    for region, spec in load_regions().items():
        if regions and region not in regions:
            continue
        df = load_region(region)
        cuts = observed_cuts(df, train_end)
        profile = ParcelaProfile(altitud=float(spec["elevation_m"]), sombra=1,
                                 humedad_tipica=float(df[HUMIDITY].loc[:train_end].mean()))
        _, R = build_environment(profile)
        for year in years:
            z, prior_B, prior_C, prior_perfect = [], [], [], []
            for m in range(1, 13):
                first = pd.Timestamp(year=year, month=m, day=1)
                origin = first - pd.Timedelta(days=1)
                obs = window_stats(df, first)                                   # clima real de la ventana (mundo)
                real = {k: tercile_index(obs[k], cuts[m][k]) for k in "THP"}
                z.append(sum(real.values()) - 3)
                prior_perfect.append(engine.model.climate_posterior(hard_evidence(real, region, m)))
                prev = origin - pd.Timedelta(days=HORIZON - 1)                  # ventana observada que ya pasó
                past = window_stats(df, prev)
                idx = {k: tercile_index(past[k], cuts[prev.month][k]) for k in "THP"}
                prior_B.append(engine.model.climate_posterior(hard_evidence(idx, region, m)))
                known = df.loc[:origin - pd.Timedelta(days=stale_days)]
                forecast = forecaster.predict(known, region, as_of=origin.date())
                prior_C.append(engine.model.climate_posterior(engine.climate(forecast, HORIZON)))
            contexts.append(Context(region=region, year=year, Q=engine.q_values(profile), R=R,
                                    pressure=rust_pressure(profile), z=z, prior_B=prior_B, prior_C=prior_C,
                                    prior_perfect=prior_perfect))
        print(f"  contexto {region}: {len(years)} años listos", flush=True)
    return contexts


def _categorical(cdf: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.minimum((u[:, None] > cdf).sum(axis=1), cdf.shape[1] - 1)


def world_matrices(world: World, cfg: dict, pressure: float) -> Dict[tuple, np.ndarray]:
    """CDF de la transición de roya por (mes anterior, z): forma (3, 5, 3)."""
    out = {}
    for month in range(1, 13):
        base = _rust_transition_wet() if month in WET_MONTHS else _rust_transition_dry()
        for z in range(-3, 4):
            out[(month, z)] = np.cumsum(_apply_pressure(base, pressure * float(np.exp(world.gamma * z))), axis=2)
    return out


def run_context(ctx: Context, model: ExtendedModel, world: World, n_rep: int, seed: int) -> List[dict]:
    cfg = model.cfg
    rng = np.random.default_rng(seed)
    u_init, u_state = rng.random(n_rep), rng.random((n_rep, 12))
    u_leaf, u_obs = rng.random((n_rep, 12, world.n_leaves)), rng.random((n_rep, 12, world.n_leaves))
    trans = world_matrices(world, cfg, ctx.pressure)
    l_cdf = np.cumsum(leaf_given_rust(cfg), axis=0)                                  # (L, R): sobre L
    o_cdf = np.cumsum(obs_given_leaf(load_confusion_counts(world.confusion), cfg["confusion"]["alpha"]), axis=1)
    log_o = np.log(model.o_given_r)                                                  # (O, R)
    w = float(cfg["leaf_weight"])
    neutral = model.climate_posterior(None)
    rows = []
    for scenario in SCENARIOS:
        r = _categorical(np.tile(np.cumsum(INITIAL), (n_rep, 1)), u_init)
        total, apps, high = np.zeros(n_rep), np.zeros(n_rep), np.zeros(n_rep)
        prev_a = np.zeros(n_rep, dtype=int)
        for m in range(1, 13):
            if m > 1:
                cdf = trans[(m - 1, ctx.z[m - 1])][r, prev_a]
                r = _categorical(cdf, u_state[:, m - 1])
            if scenario == "A_techo":
                belief = np.eye(N_STATE)[r]
            elif scenario == "B0_sin_informacion":
                belief = np.tile(neutral, (n_rep, 1))
            elif scenario == "B_clima_observado":
                belief = np.tile(ctx.prior_B[m - 1], (n_rep, 1))
            else:
                prior = {"C_pronostico": ctx.prior_C, "C0_pronostico_perfecto": ctx.prior_perfect,
                         "D_pronostico_y_CNN": ctx.prior_C, "D0_solo_CNN": None}[scenario]
                belief = np.tile(neutral if prior is None else prior[m - 1], (n_rep, 1))
                if scenario.startswith("D"):
                    leaf = np.stack([_categorical(l_cdf.T[r], u_leaf[:, m - 1, j]) for j in range(world.n_leaves)], axis=1)
                    obs = np.stack([_categorical(o_cdf[leaf[:, j]], u_obs[:, m - 1, j]) for j in range(world.n_leaves)], axis=1)
                    counts = np.stack([(obs == o).sum(axis=1) for o in range(4)], axis=1)           # (N, O)
                    belief = belief * np.exp(w * counts @ log_o)
                    belief = belief / belief.sum(axis=1, keepdims=True)
            rows_m = [state_index(m, k) for k in range(N_STATE)]
            a = (belief @ ctx.Q[rows_m]).argmax(axis=1)
            total += ctx.R[(m - 1) * N_STATE + r, a]          # state_index(m, r) para todas las réplicas
            apps += np.isin(a, (2, 3))
            high += (r == 2)
            prev_a = a
        rows.append({"region": ctx.region, "year": ctx.year, "escenario": scenario, "ingreso": float(total.mean()),
                     "aplicaciones": float(apps.mean()), "meses_roya_alta": float(high.mean())})
    return rows


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Media entre región-años y diferencia pareada contra B0 (sin información) con su error estándar."""
    pivot = table.pivot_table(index=["region", "year"], columns="escenario", values="ingreso")
    out = []
    for s in SCENARIOS:
        diff = pivot[s] - pivot["B0_sin_informacion"]
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
        apps = table[table["escenario"] == s]["aplicaciones"].mean()
        high = table[table["escenario"] == s]["meses_roya_alta"].mean()
        out.append({"escenario": s, "ingreso_medio": pivot[s].mean(), "vs_sin_informacion": diff.mean(), "error_estandar": se,
                    "aplicaciones": apps, "meses_roya_alta": high})
    ceiling = pivot["A_techo"] - pivot["B0_sin_informacion"]
    res = pd.DataFrame(out)
    res["fraccion_del_techo"] = res["vs_sin_informacion"] / ceiling.mean() if ceiling.mean() != 0 else float("nan")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--years", type=int, nargs="*", default=[2018, 2019, 2020, 2021])
    parser.add_argument("--stale", type=int, nargs="*", default=[0, 5], help="días de retraso de ERA5 al pronosticar")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--slopes", type=float, nargs="*", default=None,
                        help="pendientes de la CPT climática que cree el modelo (por defecto la de configs/fusion.yaml)")
    parser.add_argument("--gammas", type=float, nargs="*", default=None, help="fuerza del clima en el mundo simulado")
    parser.add_argument("--tag", default="", help="sufijo de los CSV de salida")
    parser.add_argument("--regions", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    forecaster = WeatherForecaster()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    if args.gammas:
        grids = [World(gamma=g) for g in args.gammas]
    else:
        grids = [World()] if not args.sensitivity else [World(g, n, c) for g in (0.0, 0.15, 0.3) for n in (5, 10, 30)
                                                        for c in ("pooled", "rocole")]
    summaries = []
    for slope in (args.slopes or [None]):
        engine = FusionEngine(load_config(None if slope is None else {"climate_risk": {"slope": slope}}))
        for stale in args.stale:
            print(f"== pendiente del modelo {engine.cfg['climate_risk']['slope']} · retraso de ERA5: {stale} días", flush=True)
            contexts = build_contexts(engine, forecaster, args.years, stale, args.regions)
            for world in grids:
                rows = []
                for i, ctx in enumerate(contexts):
                    rows += run_context(ctx, engine.model, world, args.reps, args.seed + i)
                table = pd.DataFrame(rows)
                summary = summarize(table).assign(retraso=stale, gamma=world.gamma, hojas=world.n_leaves,
                                                  confusion_mundo=world.confusion,
                                                  pendiente_modelo=engine.cfg["climate_risk"]["slope"])
                summaries.append(summary)
                if not (args.sensitivity or args.gammas):
                    table.assign(retraso=stale).round(3).to_csv(OUT / f"voi_detalle_retraso{stale}.csv", index=False)
                    print(summary.round(2).to_string(index=False))
    result = pd.concat(summaries, ignore_index=True)
    name = "voi_sensibilidad" if args.sensitivity else ("voi_pendiente_vs_mundo" if args.gammas else "voi_resumen")
    result.round(4).to_csv(OUT / f"{name}{args.tag}.csv", index=False)
    if args.sensitivity or args.gammas:
        key = result[result["escenario"].isin(["C0_pronostico_perfecto", "C_pronostico", "D0_solo_CNN", "D_pronostico_y_CNN"])]
        print(key.pivot_table(index=["pendiente_modelo", "retraso", "gamma", "hojas", "confusion_mundo"], columns="escenario",
                              values="vs_sin_informacion").round(1).to_string())


if __name__ == "__main__":
    main()

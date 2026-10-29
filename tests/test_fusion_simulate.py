"""Pruebas de la simulación del valor de la información y de la tabla de evidencia (sin pesos ni internet)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fusion.evidence import EVIDENCE_PATH, EvidenceTable
from src.fusion.network_ext import ExtendedModel
from src.fusion.qmdp import solve_parcel
from src.fusion.simulate import (SCENARIOS, Context, World, observed_cuts, run_context, summarize, tercile_index,
                                 world_matrices)
from src.mdp.environment import ParcelaProfile, build_environment, rust_pressure

PROFILE = ParcelaProfile(altitud=1400, sombra=1, humedad_tipica=75)


def make_context(model: ExtendedModel, z: int = 0) -> Context:
    _, R = build_environment(PROFILE)
    neutral = model.climate_posterior(None)
    return Context(region="x", year=2000, Q=solve_parcel(PROFILE).Q, R=R, pressure=rust_pressure(PROFILE), z=[z] * 12,
                   prior_B=[neutral] * 12, prior_C=[neutral] * 12, prior_perfect=[neutral] * 12)


def test_world_transitions_are_distributions_and_gamma_zero_is_the_inherited_mdp():
    model = ExtendedModel.build()
    for gamma in (0.0, 0.3):
        for cdf in world_matrices(World(gamma=gamma), model.cfg, 1.0).values():
            assert np.allclose(cdf[..., -1], 1.0) and (np.diff(cdf, axis=2) >= -1e-12).all()
    flat = world_matrices(World(gamma=0.0), model.cfg, 1.0)
    assert np.allclose(flat[(5, -3)], flat[(5, 3)])                       # sin efecto del clima, z no importa
    hot = world_matrices(World(gamma=0.3), model.cfg, 1.0)
    assert hot[(5, 3)][0, 0, 0] < hot[(5, -3)][0, 0, 0]                    # más clima favorable → menos probable seguir en "bajo"


def test_perfect_information_never_loses_to_no_information_and_runs_are_reproducible():
    model = ExtendedModel.build()
    ctx = make_context(model)
    rows = pd.DataFrame(run_context(ctx, model, World(gamma=0.0, n_leaves=5), n_rep=1500, seed=1))
    income = rows.set_index("escenario")["ingreso"]
    assert set(income.index) == set(SCENARIOS)
    assert income["A_techo"] >= income["B0_sin_informacion"] - 1.0
    again = pd.DataFrame(run_context(ctx, model, World(gamma=0.0, n_leaves=5), n_rep=1500, seed=1))
    pd.testing.assert_frame_equal(rows, again)


def test_summary_pairs_scenarios_with_the_no_information_baseline():
    rows = []
    for region in "ab":
        for scenario, delta in zip(SCENARIOS, (100, 0, 5, 10, 20, 30, 40)):
            rows.append({"region": region, "year": 1, "escenario": scenario, "ingreso": 1000 + delta + (7 if region == "b" else 0),
                         "aplicaciones": 8.0, "meses_roya_alta": 1.0})
    out = summarize(pd.DataFrame(rows)).set_index("escenario")
    assert out.loc["B0_sin_informacion", "vs_sin_informacion"] == 0
    assert out.loc["C_pronostico", "vs_sin_informacion"] == pytest.approx(10)       # el desfase por región se cancela al parear
    assert out.loc["D_pronostico_y_CNN", "fraccion_del_techo"] == pytest.approx(0.3)


def test_observed_cuts_are_ordered_terciles():
    idx = pd.date_range("1990-01-01", "2010-12-31", freq="D")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"temperature_2m_mean": rng.normal(20, 2, len(idx)), "relative_humidity_2m_mean": rng.normal(80, 5, len(idx)),
                       "precipitation_sum": rng.gamma(1, 5, len(idx))}, index=idx)
    cuts = observed_cuts(df, pd.Timestamp("2005-12-31"))
    assert set(cuts) == set(range(1, 13))
    for month in cuts.values():
        for lo, hi in month.values():
            assert lo < hi
    assert [tercile_index(v, (1.0, 2.0)) for v in (0.5, 1.5, 2.5)] == [0, 1, 2]


@pytest.mark.skipif(not EVIDENCE_PATH.exists(), reason="falta models/fusion/evidence.json (python -m src.fusion.build_evidence)")
def test_real_evidence_table_is_consistent():
    table = EvidenceTable.load()
    assert len(table.regions) == 6
    for region in table.regions:
        for h in ("7", "14"):
            for month in range(1, 13):
                for var, (lo, hi) in table.cuts(region, int(h), month).items():
                    assert lo < hi, (region, h, month, var)
    for h, sigmas in table.sigma.items():
        assert all(v > 0 for v in sigmas.values())

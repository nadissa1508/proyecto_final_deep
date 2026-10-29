"""Pruebas de la integración: evidencia, red bayesiana extendida y QMDP (no dependen de pesos ni de datos descargados)."""

from __future__ import annotations

import numpy as np
import pytest

from src.fusion.config import load_config
from src.fusion.contracts import ClimateEvidence, ClimateWindow
from src.fusion.evidence import tercile_probabilities
from src.fusion.network_ext import (ExtendedModel, climate_risk_cpt, leaf_given_rust, obs_given_leaf, rogan_gladen,
                                    sensitivity_specificity, wilson_interval)
from src.fusion.qmdp import qmdp, solve_parcel
from src.mdp.environment import state_index

WINDOW = ClimateWindow(region="coban", month=6, horizon_days=7, mean_temperature=19.0, mean_humidity=88.0, total_rain=60.0)
COUNTS = np.array([[40, 1, 0, 0], [0, 45, 3, 0], [0, 3, 30, 0], [0, 2, 0, 115]], dtype=float)


def evidence(pT, pH, pP) -> ClimateEvidence:
    return ClimateEvidence(window=WINDOW, p_T=pT, p_H=pH, p_P=pP)


@pytest.fixture(scope="module")
def model() -> ExtendedModel:
    cfg = load_config()
    return ExtendedModel(
        cfg=cfg, r_given_thp=climate_risk_cpt(cfg), l_given_r=leaf_given_rust(cfg), o_given_l=obs_given_leaf(COUNTS, 1.0),
        o_given_r=obs_given_leaf(COUNTS, 1.0).T @ leaf_given_rust(cfg),
        y_given_r=ExtendedModel.build().y_given_r, sensitivity=sensitivity_specificity(COUNTS)[0],
        specificity=sensitivity_specificity(COUNTS)[1])


def test_cpts_are_distributions(model):
    assert np.allclose(model.r_given_thp.sum(axis=0), 1.0)
    assert np.allclose(model.l_given_r.sum(axis=0), 1.0)
    assert np.allclose(model.o_given_l.sum(axis=1), 1.0)
    assert np.allclose(model.o_given_r.sum(axis=0), 1.0)
    assert np.allclose(model.y_given_r.sum(axis=0), 1.0)


def test_climate_risk_is_monotone_and_anchored_to_the_config(model):
    cfg = load_config()
    neutral = model.r_given_thp[:, 1, 1, 1]
    assert neutral[1] + neutral[2] == pytest.approx(cfg["climate_risk"]["neutral_at_least_medium"])
    assert neutral[2] == pytest.approx(cfg["climate_risk"]["neutral_high"])
    # más favorable (T, H, P más altos) → más probabilidad de nivel alto, y menos de bajo
    assert model.r_given_thp[2, 2, 2, 2] > neutral[2] > model.r_given_thp[2, 0, 0, 0]
    assert model.r_given_thp[0, 0, 0, 0] > neutral[0] > model.r_given_thp[0, 2, 2, 2]


def test_favorable_forecast_raises_risk_and_cold_dry_lowers_it(model):
    warm_wet = model.climate_posterior(evidence((0, 0, 1), (0, 0, 1), (0, 0, 1)))
    cold_dry = model.climate_posterior(evidence((1, 0, 0), (1, 0, 0), (1, 0, 0)))
    neutral = model.climate_posterior(None)
    assert warm_wet[2] > neutral[2] > cold_dry[2] and warm_wet[0] < neutral[0] < cold_dry[0]


def test_fast_belief_matches_the_network_with_observation_nodes(model):
    climate = evidence((0.2, 0.5, 0.3), (0.1, 0.3, 0.6), (0.3, 0.4, 0.3))
    for leaves in ([], [0], [1, 0, 0, 3], [2, 1, 1, 0, 3, 3, 0, 0, 1, 2]):
        exact = model.belief_exact(climate, leaves)
        fast = np.array(model.belief(climate, leaves).posterior)
        assert np.allclose(exact, fast, atol=1e-12)


def test_photos_move_the_belief_in_the_right_direction(model):
    climate = evidence((0.3, 0.4, 0.3), (0.3, 0.4, 0.3), (0.3, 0.4, 0.3))
    base = model.belief(climate, [])
    rusty = model.belief(climate, [1, 2, 1, 0, 1, 2, 0, 1, 1, 0])
    healthy = model.belief(climate, [0] * 10)
    assert rusty.posterior[2] > base.posterior[2] > healthy.posterior[2]
    assert healthy.posterior[0] > base.posterior[0]
    assert base.posterior == pytest.approx(base.climate_only)          # sin fotos no cambia nada


def test_leaf_weight_below_one_dampens_the_photos():
    full, soft = ExtendedModel.build(), ExtendedModel.build(load_config({"leaf_weight": 0.3}))
    leaves = [1, 2, 1, 1, 2, 1, 1, 1]
    assert soft.belief(None, leaves).posterior[2] < full.belief(None, leaves).posterior[2]


def test_pgmpy_agrees_with_variable_elimination(model):
    pytest.importorskip("pgmpy")
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    try:
        from pgmpy.models import DiscreteBayesianNetwork as Net
    except ImportError:                                  # versiones anteriores
        from pgmpy.models import BayesianNetwork as Net
    climate = evidence((0.2, 0.5, 0.3), (0.1, 0.3, 0.6), (0.3, 0.4, 0.3))
    leaves = [1, 0, 2, 3]
    net = Net([("T", "R"), ("H", "R"), ("P", "R")] + [("R", f"O{i}") for i in range(len(leaves))])
    cpds = [TabularCPD("T", 3, [[x] for x in climate.p_T]), TabularCPD("H", 3, [[x] for x in climate.p_H]),
            TabularCPD("P", 3, [[x] for x in climate.p_P]),
            TabularCPD("R", 3, model.r_given_thp.reshape(3, 27), evidence=["T", "H", "P"], evidence_card=[3, 3, 3])]
    cpds += [TabularCPD(f"O{i}", 4, model.o_given_r, evidence=["R"], evidence_card=[3]) for i in range(len(leaves))]
    net.add_cpds(*cpds)
    assert net.check_model()
    result = VariableElimination(net).query(["R"], evidence={f"O{i}": o for i, o in enumerate(leaves)}, show_progress=False)
    assert np.allclose(result.values, model.belief_exact(climate, leaves), atol=1e-9)


def test_tercile_probabilities():
    assert sum(tercile_probabilities(10.0, (8.0, 12.0), 2.0)) == pytest.approx(1.0)
    low, mid, high = tercile_probabilities(10.0, (8.0, 12.0), 2.0)
    assert mid > low and mid > high and low == pytest.approx(high)
    assert tercile_probabilities(5.0, (8.0, 12.0), 0.0) == (1.0, 0.0, 0.0)
    assert tercile_probabilities(20.0, (8.0, 12.0), 0.5)[2] > 0.99


def test_incidence_statistics():
    low, high = wilson_interval(1, 10)                   # valores citados en docs/decision_umbral_roya.md
    assert (low, high) == pytest.approx((0.018, 0.404), abs=0.002)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert rogan_gladen(0.2, 1.0, 1.0) == pytest.approx(0.2)
    assert rogan_gladen(0.0, 0.8, 0.9) == 0.0 and rogan_gladen(1.0, 0.8, 0.9) == 1.0         # recortada a [0, 1]
    assert rogan_gladen(0.3, 0.8, 0.9) == pytest.approx((0.3 + 0.9 - 1) / (0.8 + 0.9 - 1))


def test_model_incidence_uses_the_measured_detection_rates(model):
    est = model.incidence([1, 2, 0, 0, 3, 0, 0, 0, 0, 0])
    assert est.n_leaves == 10 and est.n_rust_predicted == 2 and est.apparent == pytest.approx(0.2)
    assert 0.0 <= est.low <= est.corrected <= est.high <= 1.0
    sens, spec = sensitivity_specificity(COUNTS)
    assert est.sensitivity == pytest.approx(sens) and est.specificity == pytest.approx(spec)


def test_qmdp_matches_value_iteration_for_a_known_state_and_mixes_beliefs():
    sol = solve_parcel()
    for r in range(3):
        one_hot = [1.0 if i == r else 0.0 for i in range(3)]
        result = qmdp(one_hot, 7, sol.Q)
        assert result.action == int(sol.policy[state_index(7, r)])
        assert result.values == pytest.approx(sol.Q[state_index(7, r)])
    mixed = qmdp([0.2, 0.3, 0.5], 7, sol.Q)
    expected = 0.2 * sol.Q[state_index(7, 0)] + 0.3 * sol.Q[state_index(7, 1)] + 0.5 * sol.Q[state_index(7, 2)]
    assert mixed.values == pytest.approx(expected)


def test_qmdp_rejects_invalid_inputs():
    Q = solve_parcel().Q
    with pytest.raises(ValueError):
        qmdp([0.5, 0.5, 0.5], 3, Q)
    with pytest.raises(ValueError):
        qmdp([1.0, 0.0, 0.0], 13, Q)

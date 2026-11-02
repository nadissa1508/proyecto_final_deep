"""Condiciones favorables para la roya y alarma preventiva (regla de docs/decision_umbral_roya.md §1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.fusion.favorability import FavorabilityConfig, favorable_days, preventive_alert, window_event

LITERAL = FavorabilityConfig.from_config(profile="literal")
ALTURA = FavorabilityConfig.from_config(profile="altura")
WARM_WET = ([19.0] * 7, [90.0] * 7)     # favorable con el perfil de altura, no con el literal
DRY = ([19.0] * 7, [60.0] * 7)


def test_profiles_differ_only_in_the_temperature_floor():
    assert (LITERAL.t_lo, LITERAL.t_hi) == (21.0, 25.0)
    assert ALTURA.t_lo < LITERAL.t_lo and ALTURA.t_hi == LITERAL.t_hi
    assert ALTURA.humidity_min == LITERAL.humidity_min == 85.0


def test_favorable_day_needs_temperature_range_and_humidity():
    t, h = [19, 22, 22, 26, 22], [90, 90, 80, 90, 85]
    assert favorable_days(t, h, LITERAL).tolist() == [False, True, False, False, False]   # 85 no es mayor a 85
    assert favorable_days(t, h, ALTURA).tolist() == [True, True, False, False, False]


def test_window_event_requires_the_minimum_number_of_days():
    t = [19.0] * 7
    one = [90.0] + [60.0] * 6
    two = [90.0, 90.0] + [60.0] * 5
    assert not window_event(t, one, ALTURA) and window_event(t, two, ALTURA)


def test_detected_rust_always_gives_an_alert_whatever_the_climate():
    rust = [1, 2, 0, 0]
    for temp, hum in (WARM_WET, DRY):
        for month in (1, 7):
            alert = preventive_alert(temp, hum, month=month, leaf_labels=rust, cfg=ALTURA)
            assert alert.level == "alerta" and alert.rust == "detectada"


def test_climate_only_changes_the_risk_and_the_message():
    rust = [1, 2, 0, 0]
    high = preventive_alert(*WARM_WET, month=7, leaf_labels=rust, cfg=ALTURA)
    moderate = preventive_alert(*WARM_WET, month=1, leaf_labels=rust, cfg=ALTURA)    # favorable fuera de la época
    low = preventive_alert(*DRY, month=7, leaf_labels=rust, cfg=ALTURA)
    assert (high.risk, moderate.risk, low.risk) == ("alto", "moderado", "bajo")
    assert "pronto" in high.message and "no descarte" in low.message


def test_one_rust_leaf_is_a_suspicion_not_an_alert():
    alert = preventive_alert(*WARM_WET, month=7, leaf_labels=[1, 0, 0, 0], cfg=ALTURA)
    assert alert.level == "sospecha" and alert.rust == "sospecha" and alert.rust_leaves == 1


def test_without_rust_only_high_climate_risk_asks_for_vigilance():
    assert preventive_alert(*WARM_WET, month=7, leaf_labels=[0] * 10, cfg=ALTURA).level == "vigilancia"
    assert preventive_alert(*WARM_WET, month=7, leaf_labels=[], cfg=ALTURA).rust == "sin_fotos"
    assert preventive_alert(*DRY, month=7, leaf_labels=[0] * 10, cfg=ALTURA).level == "seguimiento"
    assert preventive_alert(*WARM_WET, month=1, leaf_labels=[0] * 10, cfg=ALTURA).level == "seguimiento"   # riesgo moderado


def test_moderate_risk_with_a_single_favorable_day_in_season():
    t, h = [19.0] * 7, [90.0] + [60.0] * 6
    assert preventive_alert(t, h, month=7, leaf_labels=[], cfg=ALTURA).risk == "moderado"


def test_unreliable_forecast_is_reported_but_does_not_change_the_level():
    a = preventive_alert(*WARM_WET, month=7, leaf_labels=[1, 1], cfg=ALTURA, unreliable=True)
    b = preventive_alert(*WARM_WET, month=7, leaf_labels=[1, 1], cfg=ALTURA, unreliable=False)
    assert a.level == b.level == "alerta" and a.unreliable and any("orientativo" in r for r in a.reasons)


def test_only_the_first_window_days_count():
    t = np.array([19.0] * 14)
    h = np.array([60.0] * 7 + [95.0] * 7)             # los días favorables caen después de la ventana de 7 días
    assert preventive_alert(t, h, month=7, leaf_labels=[1, 1], cfg=ALTURA).favorable_days == 0

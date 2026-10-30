"""Pruebas de los endpoints de visión, pronóstico y decisión con módulos sintéticos (no necesitan pesos ni internet)."""

from __future__ import annotations

import io
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api import state as api_state
from src.api.main import app
from src.forecast.infer import Forecast
from src.fusion.evidence import EvidenceTable
from src.fusion.pipeline import FusionEngine
from src.vision.infer import LeafPrediction

CLASSES = ["sana", "roya_leve", "roya_severa", "otra_afeccion"]
TARGETS = ["temperature_2m_mean", "relative_humidity_2m_mean", "precipitation_sum"]


class FakeClassifier:
    classes = CLASSES

    def __init__(self, label: int = 1) -> None:
        self.label = label

    def predict(self, images, with_heatmap=False):
        probs = [0.05] * 4
        probs[self.label] = 0.85
        return [LeafPrediction(label=self.label, label_name=CLASSES[self.label], confidence=0.85, probs=probs,
                               per_model={"m1": probs}, agree=True,
                               heatmap_png_base64="AAAA" if with_heatmap else None) for _ in images]


class DoubtfulClassifier(FakeClassifier):
    """La segunda foto es no concluyente: el resto se comporta igual que FakeClassifier."""

    def predict(self, images, with_heatmap=False):
        out = super().predict(images, with_heatmap)
        if len(out) > 1:
            out[1].usable, out[1].reason = False, "confianza baja (30%)"
        return out


class FakeForecaster:
    def predict(self, history, region, as_of=None):
        n = 14
        days = [(pd.Timestamp(as_of) + pd.Timedelta(days=h)).date().isoformat() for h in range(1, n + 1)]
        return Forecast(region_used=region, last_observed=history.index[-1].date().isoformat(), as_of=str(as_of),
                        stale_days=int((pd.Timestamp(as_of) - history.index[-1]).days), days=days,
                        values={TARGETS[0]: [19.0] * n, TARGETS[1]: [88.0] * n, TARGETS[2]: [8.0] * n},
                        spread={t: [0.1] * n for t in TARGETS}, source={t: "base" for t in TARGETS},
                        degraded=[False] * n, notes=[])


def synthetic_table() -> EvidenceTable:
    cuts = {t: [10.0 if t != "P" else 50.0, 20.0 if t != "P" else 100.0] for t in "THP"}
    cuts["H"] = [80.0, 90.0]
    thresholds = {"coban": {"14": {str(m): cuts for m in range(1, 13)}}}
    return EvidenceTable(thresholds, {"14": {"T": 0.7, "H": 5.0, "P": 1.0}}, {})


def png(color=(40, 160, 60), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client(monkeypatch):
    state = api_state.get_state()
    saved = (state.vision, state.forecaster, state.fusion)
    state.vision, state.forecaster, state.fusion = FakeClassifier(), FakeForecaster(), FusionEngine(table=synthetic_table())
    history = pd.DataFrame({"relative_humidity_2m_mean": np.full(100, 85.0)},
                           index=pd.date_range(end=pd.Timestamp(date.today()) - pd.Timedelta(days=6), periods=100, freq="D"))
    monkeypatch.setattr("src.api.routers.forecast.load_history", lambda region, source: (history, "prueba", []))
    monkeypatch.setattr("src.api.routers.decision.load_history", lambda region, source: (history, "prueba", []))
    yield TestClient(app)
    state.vision, state.forecaster, state.fusion = saved


def test_vision_predict_returns_one_result_per_photo(client):
    files = [("files", ("a.png", png(), "image/png")), ("files", ("b.png", png((10, 10, 10)), "image/png"))]
    body = client.post("/api/vision/predict?heatmap=true", files=files).json()
    assert len(body["hojas"]) == 2 and body["hojas"][0]["clase"] == "roya_leve"
    assert sum(body["hojas"][0]["probabilidades"].values()) == pytest.approx(1.0)
    assert body["hojas"][0]["mapa_calor_png_base64"] == "AAAA" and body["advertencia"]


def test_vision_rejects_bad_uploads(client):
    assert client.post("/api/vision/predict", files=[("files", ("a.gif", b"GIF89a", "image/gif"))]).status_code == 415
    assert client.post("/api/vision/predict", files=[("files", ("a.png", b"no soy imagen", "image/png"))]).status_code == 422
    assert client.post("/api/vision/predict", files=[("files", ("a.heic", b"x", "image/heic"))]).status_code == 415
    assert client.post("/api/vision/predict").status_code == 422
    big = png(size=(8, 8)) + b"\0" * (10 * 1024 * 1024)
    assert client.post("/api/vision/predict", files=[("files", ("g.png", big, "image/png"))]).status_code == 413


def test_modules_missing_answer_503_not_500(client):
    state = api_state.get_state()
    state.vision, state.vision_error = None, "faltan pesos"
    r = client.post("/api/vision/predict", files=[("files", ("a.png", png(), "image/png"))])
    assert r.status_code == 503 and "faltan pesos" in r.json()["detail"]
    state.forecaster, state.forecast_error = None, "sin artefacto"
    assert client.get("/api/forecast/coban").status_code == 503


def test_forecast_endpoint(client):
    ok = client.get("/api/forecast/coban?source=local")
    assert ok.status_code == 200
    body = ok.json()
    assert body["dias_de_retraso"] == 6 and len(body["dias"]) == 14 and set(body["valores"]) == set(TARGETS)
    assert client.get("/api/forecast/atlantida").status_code == 404
    assert client.get("/api/forecast/coban?source=otra").status_code == 422


def test_decision_with_and_without_photos(client):
    without = client.post("/api/decision", data={"region": "coban"}).json()
    assert without["incidencia"] is None and without["accion"] in without["valor_esperado_por_accion"]
    assert sum(without["creencia"].values()) == pytest.approx(1.0, abs=1e-3)
    assert without["creencia"] == without["creencia_solo_clima"]
    files = [("files", (f"h{i}.png", png(), "image/png")) for i in range(12)]
    with_photos = client.post("/api/decision", data={"region": "coban", "altitud": "1300"}, files=files).json()
    assert with_photos["incidencia"]["hojas"] == 12 and with_photos["incidencia"]["hojas_con_roya"] == 12
    assert with_photos["creencia"]["alto"] > without["creencia"]["alto"]            # 12 hojas con roya suben el riesgo
    few = client.post("/api/decision", data={"region": "coban"}, files=files[:3]).json()
    assert any("al menos" in n for n in few["notas"])


def test_decision_validates_inputs(client):
    assert client.post("/api/decision", data={"region": "atlantida"}).status_code == 404
    assert client.post("/api/decision", data={"region": "coban", "sombra": "7"}).status_code == 422
    assert client.post("/api/decision", data={"region": "coban", "source": "x"}).status_code == 422


def test_legacy_endpoints_are_untouched(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_doubtful_photos_are_reported_and_not_used_in_the_decision(client):
    api_state.get_state().vision = DoubtfulClassifier()
    files = [("files", ("buena.png", png(), "image/png")), ("files", ("dudosa.png", png((5, 5, 5)), "image/png"))]
    vision = client.post("/api/vision/predict", files=files).json()
    assert [h["concluyente"] for h in vision["hojas"]] == [True, False]
    assert "confianza baja" in vision["hojas"][1]["motivo"]

    body = client.post("/api/decision", data={"region": "coban", "source": "local"}, files=files).json()
    assert body["hojas_descartadas"] == 1 and len(body["hojas"]) == 1
    assert any("dudosa.png" in n and "no ser concluyentes" in n for n in body["notas"])

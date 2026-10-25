"""La interfaz de inferencia debe calibrar, combinar modelos y fallar con un error claro si no hay pesos."""

import base64
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.vision.infer import LeafClassifier, ModelNotAvailable
from src.vision.models import build_model

CLASSES = ["sana", "roya_leve", "roya_severa", "otra_afeccion"]


def _artifact(tmp_path, temperatures=(1.0,)):
    entries = []
    for i, t in enumerate(temperatures):
        torch.manual_seed(i)
        weights = tmp_path / f"m{i}.pt"
        torch.save(build_model("resnet18", "full_dlr", pretrained=False).state_dict(), weights)
        entries.append({"name": f"m{i}", "arch": "resnet18", "weights": str(weights), "temperature": t})
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({"classes": CLASSES, "img_size": 64, "models": entries}), encoding="utf-8")
    return path


def _image(seed=0):
    return Image.fromarray(np.random.RandomState(seed).randint(0, 255, (50, 70, 3), dtype=np.uint8))


def test_sin_artefacto_lanza_model_not_available(tmp_path):
    with pytest.raises(ModelNotAvailable):
        LeafClassifier(tmp_path / "no_existe.json", device="cpu")


def test_faltan_los_pesos_lanza_model_not_available(tmp_path):
    path = _artifact(tmp_path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["models"][0]["weights"] = str(tmp_path / "fantasma.pt")
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ModelNotAvailable):
        LeafClassifier(path, device="cpu")


def test_las_probabilidades_suman_uno_y_la_temperatura_solo_cambia_la_confianza(tmp_path):
    cold = LeafClassifier(_artifact(tmp_path, (1.0,)), device="cpu").predict_one(_image())
    hot_dir = tmp_path / "hot"
    hot_dir.mkdir()
    hot = LeafClassifier(_artifact(hot_dir, (5.0,)), device="cpu").predict_one(_image())
    assert sum(cold.probs) == pytest.approx(1.0) and len(cold.probs) == 4
    assert hot.label == cold.label                                  # calibrar no cambia la clase
    assert hot.confidence < cold.confidence                         # T > 1 suaviza
    assert cold.label_name == CLASSES[cold.label] and cold.agree


def test_ensamble_promedia_modelos_y_reporta_acuerdo(tmp_path):
    clf = LeafClassifier(_artifact(tmp_path, (1.0, 1.0)), device="cpu")
    pred = clf.predict_one(_image(1))
    assert set(pred.per_model) == {"m0", "m1"}
    mean = np.mean([pred.per_model["m0"], pred.per_model["m1"]], axis=0)
    assert np.allclose(pred.probs, mean)
    assert pred.agree == (int(np.argmax(pred.per_model["m0"])) == int(np.argmax(pred.per_model["m1"])))


def test_varias_imagenes_y_mapa_de_calor_en_png(tmp_path):
    clf = LeafClassifier(_artifact(tmp_path), device="cpu")
    preds = clf.predict([_image(0), _image(1), _image(2)], with_heatmap=True)
    assert len(preds) == 3
    png = Image.open(io.BytesIO(base64.b64decode(preds[0].heatmap_png_base64)))
    assert png.size == (64, 64) and png.format == "PNG"

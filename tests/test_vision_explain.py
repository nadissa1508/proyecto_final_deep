"""Grad-CAM y sus validaciones se prueban con modelos sintéticos donde la respuesta correcta se conoce."""

import numpy as np
import pytest
import torch
from torch import nn

from src.vision.explain import (GradCAM, background_attention, background_mask, cascading_randomization,
                                deletion_insertion, denormalize)
from src.vision.models import build_model


class GapNet(nn.Module):
    """conv -> ReLU -> promedio global -> lineal: arquitectura donde Grad-CAM coincide con CAM."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(self.features(x).mean(dim=(2, 3)))


def test_gradcam_coincide_con_cam_en_arquitectura_con_pooling_global():
    torch.manual_seed(0)
    model, x = GapNet().eval(), torch.randn(2, 3, 16, 16)
    with GradCAM(model, model.features) as g:
        cam, logits, idx = g(x)
    activations = model.features(x)                                   # (2, 8, 16, 16)
    for i in range(2):
        w = model.fc.weight[idx[i]]                                   # α_k = w_k (el 1/Z se absorbe al normalizar)
        expected = torch.relu((w[:, None, None] * activations[i]).sum(0)).detach()
        expected = (expected - expected.min()) / (expected.max() - expected.min() + 1e-8)
        assert torch.allclose(cam[i], expected, atol=1e-5)
    assert cam.min() >= 0 and cam.max() <= 1 + 1e-6 and (idx == logits.argmax(1)).all()


class QuadrantNet(nn.Module):
    """La clase 1 depende del contraste (desviación estándar) del cuadrante superior izquierdo.

    Se usa el contraste y no el promedio porque una imagen constante desenfocada sigue igual y la curva
    de insertion sería plana; el desenfoque sí borra la textura de un tablero de ajedrez.
    """

    def forward(self, x):
        score = x[:, :, : x.shape[2] // 2, : x.shape[3] // 2].flatten(1).std(dim=1)
        return torch.stack([torch.zeros_like(score), 6.0 * score], dim=1)


def _checkerboard(side=32):
    rows, cols = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    return (((rows + cols) % 2).float() * 2 - 1).expand(1, 3, side, side).contiguous()


def test_deletion_insertion_premian_el_mapa_correcto_frente_al_azar():
    x = _checkerboard()
    good = torch.zeros(32, 32)
    good[:16, :16] = 1.0                                              # mapa que señala el cuadrante relevante
    rows, cols = torch.meshgrid(torch.arange(32.0), torch.arange(32.0), indexing="ij")
    bad = (rows + cols) / 62.0                                        # adverso: el cuadrante relevante es lo MENOS importante
    m = QuadrantNet()
    r_good = deletion_insertion(m, x, good, class_idx=1, grid=8, n_steps=16)
    r_bad = deletion_insertion(m, x, bad, class_idx=1, grid=8, n_steps=16)
    assert r_good["deletion_auc"] < r_good["deletion_auc_aleatorio"] < r_bad["deletion_auc"]
    assert r_good["insertion_auc"] > r_good["insertion_auc_aleatorio"] > r_bad["insertion_auc"]
    assert r_good["deletion_auc"] < 0.75 * r_good["deletion_auc_aleatorio"]    # la diferencia es grande, no marginal


def test_atencion_al_fondo_distingue_hoja_de_fondo():
    image = np.ones((20, 20, 3))                                      # fondo blanco
    image[5:15, 5:15] = [0.2, 0.55, 0.2]                              # "hoja" verde
    on_leaf = np.zeros((20, 20)); on_leaf[5:15, 5:15] = 1.0
    on_background = 1.0 - on_leaf
    assert background_mask(image).mean() == pytest.approx(0.75)
    assert background_attention(image, on_leaf)["masa_en_fondo"] == 0.0
    assert background_attention(image, on_background)["cociente"] == pytest.approx(1 / 0.75)


def test_aleatorizacion_en_cascada_devuelve_una_fila_por_etapa():
    torch.manual_seed(0)
    model = build_model("resnet18", "full_dlr", pretrained=False).eval()
    rows = cascading_randomization(model, "resnet18", torch.randn(1, 3, 64, 64), class_idx=2)
    assert [name for name, _ in rows] == ["original", "+fc", "+layer4", "+layer3", "+layer2", "+layer1", "+conv1"]
    assert rows[0][1] == 1.0
    assert all(np.isnan(v) or -1.0 <= v <= 1.0 for _, v in rows)


def test_denormalize_invierte_la_normalizacion():
    from src.vision.transforms import IMAGENET_MEAN, IMAGENET_STD

    image = torch.rand(3, 8, 8)
    mean, std = torch.tensor(IMAGENET_MEAN)[:, None, None], torch.tensor(IMAGENET_STD)[:, None, None]
    assert np.allclose(denormalize((image - mean) / std), image.permute(1, 2, 0).numpy(), atol=1e-5)

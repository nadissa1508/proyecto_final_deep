"""Temperature scaling debe reducir la sobreconfianza sin cambiar las predicciones."""

import numpy as np
import torch

from src.vision.calibration import calibrate, expected_calibration_error, fit_temperature


def _overconfident(n=4000, classes=4, scale=4.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, classes, (n,), generator=g)
    logits = torch.randn(n, classes, generator=g)
    logits[torch.arange(n), labels] += 1.5          # señal real moderada
    return logits * scale, labels                   # multiplicar por `scale` lo vuelve sobreconfiado


def test_la_temperatura_corrige_la_sobreconfianza():
    logits, labels = _overconfident()
    t = fit_temperature(logits[:2000], labels[:2000])
    assert t > 1.5                                   # una red sobreconfiada necesita T > 1
    before = expected_calibration_error(torch.softmax(logits[2000:], 1).numpy(), labels[2000:].numpy())
    after = expected_calibration_error(calibrate(logits[2000:], t).numpy(), labels[2000:].numpy())
    assert after < before / 2


def test_la_calibracion_no_cambia_la_clase_predicha():
    logits, labels = _overconfident(n=500)
    t = fit_temperature(logits, labels)
    assert (calibrate(logits, t).argmax(1) == logits.argmax(1)).all()


def test_una_red_ya_calibrada_queda_con_temperatura_cercana_a_uno():
    g = torch.Generator().manual_seed(1)
    logits = 2.0 * torch.randn(20000, 4, generator=g)
    labels = torch.multinomial(torch.softmax(logits, 1), 1, generator=g).squeeze(1)  # etiquetas ~ probabilidades
    assert 0.9 < fit_temperature(logits, labels) < 1.1
    assert expected_calibration_error(torch.softmax(logits, 1).numpy(), labels.numpy()) < 0.03

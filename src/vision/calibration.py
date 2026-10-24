"""Calibración de probabilidades por temperature scaling y su medición (ECE).

Una red mal calibrada dice "97 %" y acierta mucho menos. Temperature scaling divide los logits
por un único número T > 0 (`softmax(z / T)`), ajustado sobre **validación** minimizando la
pérdida logarítmica; no cambia la clase predicha, solo la confianza. T > 1 suaviza una red
sobreconfiada.

- `fit_temperature(logits, labels)`: T óptimo (se optimiza log T para mantener T > 0).
- `expected_calibration_error(probs, labels, bins)`: ECE; promedio ponderado de
  |exactitud − confianza| por intervalo de confianza.
- `calibrate(logits, T)`: probabilidades calibradas.

El ajuste usa solo validación y el ECE se reporta en test, para no medir la calibración con los
mismos datos con los que se ajustó.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    logits, labels = logits.double(), labels.long()
    log_t = torch.zeros(1, dtype=torch.double, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def calibrate(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.softmax(logits.double() / temperature, dim=1)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)

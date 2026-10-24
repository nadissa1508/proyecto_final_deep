"""Modelos de pronóstico: LSTM, GRU y Transformer, todos de salida directa multi-horizonte.

Entrada: (lote, input_days, F) con F = variables de ERA5 normalizadas + sin/cos del día del año.
Salida: (lote, max_horizon, T): para cada uno de los próximos 14 días, las T variables objetivo
(en la escala normalizada; la lluvia va en `log1p`). Se predicen los 14 días de una sola vez,
sin realimentar predicciones, para no acumular error (S12, "direct multi-output").

- `RecurrentForecaster`: LSTM o GRU apilado; el estado oculto del último día alimenta una cabeza lineal.
- `TransformerForecaster`: proyección lineal + codificación posicional sinusoidal (S6) + encoder
  con atención multi-cabeza; la cabeza lineal recibe el último día y el promedio de todos los días.
  La codificación posicional es imprescindible: sin ella el Transformer no distingue el orden.

Se usan los bloques de PyTorch (`nn.LSTM`, `nn.GRU`, `nn.TransformerEncoder`); las versiones
escritas desde cero están en los laboratorios 2, 5 y 7 del curso.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class RecurrentForecaster(nn.Module):
    def __init__(self, kind: str, n_features: int, n_targets: int, horizon: int,
                 hidden: int = 128, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        cell = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = cell(n_features, hidden, num_layers=layers, batch_first=True,
                        dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, horizon * n_targets))
        self.horizon, self.n_targets = horizon, n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).view(-1, self.horizon, self.n_targets)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div), torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(self, n_features: int, n_targets: int, horizon: int, d_model: int = 64,
                 heads: int = 4, layers: int = 3, ff: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        block = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Dropout(dropout),
                                  nn.Linear(2 * d_model, horizon * n_targets))
        self.horizon, self.n_targets = horizon, n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.pos(self.proj(x)))
        pooled = torch.cat([h[:, -1], h.mean(dim=1)], dim=-1)
        return self.head(pooled).view(-1, self.horizon, self.n_targets)


def build_model(cfg: dict, n_features: int, n_targets: int, horizon: int) -> nn.Module:
    kind = cfg["model"]
    if kind in ("lstm", "gru"):
        return RecurrentForecaster(kind, n_features, n_targets, horizon, cfg["hidden"], cfg["layers"], cfg["dropout"])
    if kind == "transformer":
        return TransformerForecaster(n_features, n_targets, horizon, cfg["d_model"], cfg["heads"],
                                     cfg["layers"], cfg["ff"], cfg["dropout"])
    raise ValueError(f"Modelo desconocido: {kind}")

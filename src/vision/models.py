"""Modelos de visión y estrategias de transfer learning (las del Lab 6).

Modelos: `resnet18` (principal), `efficientnet_b0` (comparación en la ablación) y
`small_cnn` (baseline desde cero). Todos devuelven 4 logits (sana, roya leve, roya
severa, otra afección).

Estrategias:
- `feature_extraction`: backbone congelado; solo se entrena la cabeza nueva.
- `partial`: se descongela el último bloque (`layer4` en ResNet, `features[6:]` en
  EfficientNet) junto con la cabeza; el bloque va a `lr / 10`.
- `full_dlr`: todo entrenable con learning rate diferencial por grupo, de las capas
  tempranas (lr muy pequeño) a la cabeza (lr más grande). 6 grupos.
- `scratch`: sin pesos preentrenados (solo para `small_cnn`).

Las capas congeladas con BatchNorm se dejan en modo evaluación: si no, sus estadísticas
de ejecución se seguirían actualizando aunque los pesos estén congelados.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn
from torchvision import models

NUM_CLASSES = 4
STRATEGIES = ("feature_extraction", "partial", "full_dlr", "scratch")


class SmallCNN(nn.Module):
    """Cuatro bloques conv-BN-ReLU-pool y una capa lineal; baseline sin preentrenar."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        channels = [3, 16, 32, 64, 128]
        blocks = []
        for cin, cout in zip(channels[:-1], channels[1:]):
            blocks += [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.features(x)).flatten(1))


def build_model(name: str, strategy: str, pretrained: bool = True) -> nn.Module:
    if strategy not in STRATEGIES:
        raise ValueError(f"Estrategia desconocida: {strategy}")
    if name == "small_cnn":
        return SmallCNN()
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Modelo desconocido: {name}")
    apply_strategy(model, name, strategy)
    return model


def _head(model: nn.Module, name: str) -> nn.Module:
    return model.fc if name == "resnet18" else model.classifier


def _last_block(model: nn.Module, name: str) -> nn.Module:
    return model.layer4 if name == "resnet18" else model.features[6:]


def apply_strategy(model: nn.Module, name: str, strategy: str) -> None:
    for p in model.parameters():
        p.requires_grad = strategy in ("full_dlr", "scratch")
    if strategy in ("feature_extraction", "partial"):
        for p in _head(model, name).parameters():
            p.requires_grad = True
    if strategy == "partial":
        for p in _last_block(model, name).parameters():
            p.requires_grad = True


def param_groups(model: nn.Module, name: str, strategy: str, lr: float,
                 lr_block: float, lrs_dlr: List[float]) -> List[dict]:
    """Grupos de parámetros con su learning rate según la estrategia."""
    if strategy == "full_dlr":
        if name == "resnet18":
            groups = [[model.conv1, model.bn1], [model.layer1], [model.layer2],
                      [model.layer3], [model.layer4], [model.fc]]
        elif name == "efficientnet_b0":
            f = model.features
            groups = [[f[0], f[1]], [f[2], f[3]], [f[4], f[5]], [f[6], f[7]], [f[8]], [model.classifier]]
        else:
            raise ValueError("full_dlr requiere resnet18 o efficientnet_b0")
        if len(lrs_dlr) != len(groups):
            raise ValueError(f"lrs_dlr debe tener {len(groups)} valores")
        return [{"params": [p for m in g for p in m.parameters()], "lr": r}
                for g, r in zip(groups, lrs_dlr)]
    if strategy == "partial":
        block = list(_last_block(model, name).parameters())
        ids = {id(p) for p in block}
        head = [p for p in _head(model, name).parameters() if id(p) not in ids]
        return [{"params": block, "lr": lr_block}, {"params": head, "lr": lr}]
    return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]


def set_train_mode(model: nn.Module) -> None:
    """`model.train()` pero con las BatchNorm congeladas en modo evaluación."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in m.parameters()):
                m.eval()


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "entrenables": trainable, "congelados": total - trainable}

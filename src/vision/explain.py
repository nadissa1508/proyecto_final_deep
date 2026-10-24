"""Grad-CAM y sus validaciones: aleatorización de pesos, deletion/insertion y atención al fondo.

Grad-CAM (Selvaraju et al., 2017): para la clase c, los pesos de cada canal k de la última capa
convolucional son el promedio espacial del gradiente, α_k = (1/Z) Σ_ij ∂y_c/∂A^k_ij, y el mapa es
L = ReLU(Σ_k α_k A^k). `y_c` es el logit (antes del softmax). El mapa se reescala al tamaño de la
imagen y se normaliza a [0, 1] por imagen. Su resolución es gruesa (8×8 con entrada de 256 px).

Un mapa bonito no prueba que el modelo mire lo correcto, por eso hay tres comprobaciones:

1. `cascading_randomization` (Adebayo et al., 2018): se reinician los pesos del modelo capa por
   capa, de la salida hacia la entrada, y se mide la correlación de Spearman entre el mapa
   original y el del modelo degradado. Si el mapa casi no cambia, el método no explica al modelo.
2. `deletion_insertion` (Petsiuk et al., 2018): se quitan (deletion) o se agregan (insertion) los
   parches más importantes según el mapa y se sigue la probabilidad de la clase; se resume con el
   área bajo la curva (AUC) y se compara con un orden aleatorio. Un mapa útil hace que la
   probabilidad caiga rápido al borrar lo importante (deletion bajo) y suba rápido al agregarlo
   (insertion alto), mejor que al azar.
3. `background_attention`: fracción de la masa del mapa que cae sobre el fondo claro. Es un
   **heurístico** pensado para BRACOL (hoja sobre fondo blanco): píxeles con poca saturación y
   mucho brillo. No tiene sentido en fotos de campo.

Todo corre en CPU o GPU según el dispositivo del modelo.
"""

from __future__ import annotations

import base64
import copy
import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from torch import nn
from torchvision.transforms import functional as TF

from src.vision.transforms import IMAGENET_MEAN, IMAGENET_STD


def target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Última capa convolucional (antes del pooling global) de cada arquitectura."""
    if arch == "resnet18":
        return model.layer4
    if arch == "efficientnet_b0":
        return model.features[-1]
    if arch == "small_cnn":
        return model.features[-2]            # ReLU del último bloque, antes del max-pool
    raise ValueError(f"Arquitectura desconocida: {arch}")


class GradCAM:
    """Calcula mapas Grad-CAM para un modelo y una capa objetivo. Usar `with GradCAM(...) as cam:`."""

    def __init__(self, model: nn.Module, layer: nn.Module) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handle = layer.register_forward_hook(self._on_forward)

    def _on_forward(self, _module, _inputs, output: torch.Tensor) -> None:
        self.activations = output
        if output.requires_grad:
            output.register_hook(lambda grad: setattr(self, "gradients", grad))

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()

    def remove(self) -> None:
        self._handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Devuelve (mapas (n, H, W) en [0, 1], logits (n, C), clases usadas (n,))."""
        self.model.eval()
        x = x.clone().requires_grad_(True)           # garantiza grafo aunque el backbone esté congelado
        with torch.enable_grad():
            logits = self.model(x)
            idx = logits.argmax(1) if class_idx is None else class_idx.to(logits.device)
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, idx[:, None]).sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[:, 0]
        flat = cam.flatten(1)
        lo, hi = flat.min(1).values, flat.max(1).values
        cam = (cam - lo[:, None, None]) / (hi - lo)[:, None, None].clamp_min(1e-8)
        return cam.detach(), logits.detach(), idx.detach()


# --------------------------------------------------------------------------- aleatorización de pesos

def _stages(model: nn.Module, arch: str) -> List[Tuple[str, List[nn.Module]]]:
    """Etapas en el orden en que se reinician: de la salida hacia la entrada."""
    if arch == "resnet18":
        return [("fc", [model.fc]), ("layer4", [model.layer4]), ("layer3", [model.layer3]),
                ("layer2", [model.layer2]), ("layer1", [model.layer1]), ("conv1", [model.conv1, model.bn1])]
    if arch == "efficientnet_b0":
        return [("classifier", [model.classifier])] + [(f"features.{i}", [model.features[i]]) for i in range(8, -1, -1)]
    raise ValueError(f"Sin etapas definidas para {arch}")


def _reset(module: nn.Module) -> None:
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().cpu().numpy(), b.flatten().cpu().numpy()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def cascading_randomization(model: nn.Module, arch: str, x: torch.Tensor, class_idx: int,
                            seed: int = 0) -> List[Tuple[str, float]]:
    """Correlación de Spearman entre el mapa original y el del modelo con las últimas etapas reiniciadas.

    `x` es una sola imagen (1, 3, H, W). Se fija la clase del mapa original para que la comparación
    sea sobre lo mismo. Si el método explica al modelo, la correlación debe bajar hacia 0 al ir
    reiniciando etapas. El orden de las filas es de la salida hacia la entrada, acumulativo.
    """
    idx = torch.tensor([class_idx], device=x.device)
    with GradCAM(model, target_layer(model, arch)) as g:
        original, _, _ = g(x, idx)
    degraded = copy.deepcopy(model)
    rows = [("original", 1.0)]
    torch.manual_seed(seed)
    for name, modules in _stages(degraded, arch):
        for m in modules:
            _reset(m)
        with GradCAM(degraded, target_layer(degraded, arch)) as g:
            cam, _, _ = g(x, idx)
        rows.append((f"+{name}", _spearman(original, cam)))
    return rows


# --------------------------------------------------------------------------- deletion / insertion

def _auc(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum((y[1:] + y[:-1]) * (x[1:] - x[:-1]) / 2.0))


@torch.no_grad()
def _class_probability(model: nn.Module, batch: torch.Tensor, class_idx: int, chunk: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(batch), chunk):
        out.append(F.softmax(model(batch[i:i + chunk]), dim=1)[:, class_idx].cpu())
    return torch.cat(out).numpy()


def _curve(model: nn.Module, x: torch.Tensor, order: np.ndarray, grid: int, n_steps: int,
           class_idx: int, mode: str, blurred: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    patch = x.shape[-1] // grid
    total = grid * grid
    fractions = np.linspace(0.0, 1.0, n_steps + 1)
    images = []
    for frac in fractions:
        revealed = torch.zeros(total, dtype=torch.bool)
        revealed[torch.from_numpy(order[: int(round(frac * total))].copy())] = True
        pixel_mask = revealed.view(grid, grid).repeat_interleave(patch, 0).repeat_interleave(patch, 1)
        pixel_mask = pixel_mask.to(x.device)[None, None]
        images.append(torch.where(pixel_mask, torch.zeros_like(x), x)[0] if mode == "deletion"
                      else torch.where(pixel_mask, x, blurred)[0])
    return fractions, _class_probability(model, torch.stack(images), class_idx)


def deletion_insertion(model: nn.Module, x: torch.Tensor, cam: torch.Tensor, class_idx: int,
                       grid: int = 16, n_steps: int = 32, n_random: int = 3, seed: int = 0) -> Dict[str, float]:
    """AUC de las curvas de deletion e insertion según el mapa, y su promedio con orden aleatorio.

    `x` es (1, 3, H, W) normalizada y `cam` (H, W) en [0, 1]. Deletion pone a cero (el color medio, ya
    que la imagen está normalizada) los parches más importantes primero; insertion parte de una versión
    borrosa y va revelando los parches más importantes. Se mide la probabilidad de `class_idx`.
    """
    model.eval()
    side = x.shape[-1]
    if side % grid:
        raise ValueError("El tamaño de la imagen debe ser múltiplo de `grid`")
    patch = side // grid
    importance = cam.to(x.device).reshape(grid, patch, grid, patch).mean(dim=(1, 3)).flatten().cpu().numpy()
    order = np.argsort(-importance, kind="stable")
    blurred = TF.gaussian_blur(x, kernel_size=[31, 31], sigma=[8.0, 8.0])
    fractions, deletion = _curve(model, x, order, grid, n_steps, class_idx, "deletion", blurred)
    _, insertion = _curve(model, x, order, grid, n_steps, class_idx, "insertion", blurred)
    rng = np.random.RandomState(seed)
    rand_del, rand_ins = [], []
    for _ in range(n_random):
        shuffled = rng.permutation(grid * grid)
        rand_del.append(_auc(_curve(model, x, shuffled, grid, n_steps, class_idx, "deletion", blurred)[1], fractions))
        rand_ins.append(_auc(_curve(model, x, shuffled, grid, n_steps, class_idx, "insertion", blurred)[1], fractions))
    return {"deletion_auc": _auc(deletion, fractions), "insertion_auc": _auc(insertion, fractions),
            "deletion_auc_aleatorio": float(np.mean(rand_del)), "insertion_auc_aleatorio": float(np.mean(rand_ins))}


# --------------------------------------------------------------------------- fondo y visualización

def denormalize(x: torch.Tensor) -> np.ndarray:
    """Tensor normalizado (3, H, W) o (1, 3, H, W) -> imagen RGB en [0, 1], forma (H, W, 3)."""
    x = x[0] if x.dim() == 4 else x
    mean = torch.tensor(IMAGENET_MEAN, device=x.device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, device=x.device)[:, None, None]
    return (x * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def background_mask(image01: np.ndarray, max_saturation: float = 0.15, min_value: float = 0.7) -> np.ndarray:
    """Heurístico de fondo claro y gris/blanco (hoja sobre papel): poca saturación y mucho brillo."""
    high, low = image01.max(axis=2), image01.min(axis=2)
    saturation = (high - low) / np.maximum(high, 1e-6)
    return (saturation < max_saturation) & (high > min_value)


def background_attention(image01: np.ndarray, cam: np.ndarray) -> Dict[str, float]:
    """Masa del mapa sobre el fondo, fracción de área del fondo y su cociente.

    Un cociente cercano a 1 significa que el mapa reparte su atención como el área; menor que 1, que
    evita el fondo; mayor que 1, que lo prefiere (señal de alarma de un atajo de adquisición).
    """
    mask = background_mask(image01)
    area, total = float(mask.mean()), float(cam.sum())
    mass = float(cam[mask].sum() / total) if total > 0 else float("nan")
    return {"masa_en_fondo": mass, "area_fondo": area, "cociente": mass / area if area > 0 else float("nan")}


def overlay_png_base64(image01: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> str:
    """Superpone el mapa (colormap `jet`) sobre la imagen y lo devuelve como PNG en base64."""
    import matplotlib

    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    blend = np.clip((1 - alpha) * image01 + alpha * heat, 0, 1)
    buffer = io.BytesIO()
    Image.fromarray((blend * 255).astype(np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")

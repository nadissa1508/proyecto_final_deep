"""Valida el Grad-CAM de los modelos reales: aleatorización de pesos, deletion/insertion y fondo.

Para cada modelo toma una muestra estratificada del test de BRACOL y del de RoCoLe (`--per-class` imágenes
por clase), calcula el mapa de la clase que el propio modelo predice y mide:

1. aleatorización en cascada: correlación de Spearman entre el mapa original y el del modelo con las últimas
   etapas reiniciadas (debe caer hacia 0);
2. deletion e insertion frente a un orden aleatorio de parches (deletion debe ser menor que el aleatorio e
   insertion mayor);
3. solo en BRACOL: cociente entre la masa del mapa sobre el fondo claro y el área que ocupa el fondo
   (< 1 evita el fondo, > 1 lo prefiere). Es un heurístico.

Se guardan los valores por imagen en `experiments/vision/gradcam_validation.csv`, el resumen en
`experiments/vision/gradcam_summary.json` y una figura de ejemplos en `docs/figures/gradcam_ejemplos_<modelo>.png`.

Uso:
    python -m src.vision.validate_explanations
    python -m src.vision.validate_explanations --per-class 4
"""

from __future__ import annotations

import argparse
import json
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.common.paths import EXPERIMENTS_DIR, MODELS_DIR, ROOT
from src.vision.datasets import CLASSES
from src.vision.explain import (GradCAM, background_attention, cascading_randomization, deletion_insertion,
                                denormalize, target_layer)
from src.vision.models import build_model
from src.vision.splits import SPLITS_PATH
from src.vision.train import LeafDataset
from src.vision.transforms import eval_transform

IMG_SIZE = 256
SHORT = ["sana", "leve", "severa", "otra"]
MODELS = [("resnet18_dlr_both", "resnet18"), ("efficientnet_b0_dlr_both", "efficientnet_b0")]


def sample(splits: pd.DataFrame, dataset: str, per_class: int, seed: int = 0) -> pd.DataFrame:
    test = splits[(splits["dataset"] == dataset) & (splits["split"] == "test")]
    parts = [g.sample(min(per_class, len(g)), random_state=seed) for _, g in test.groupby("label")]
    return pd.concat(parts).sort_values("label").reset_index(drop=True)


def center_prior(device: torch.device, side: int = IMG_SIZE, sigma: float = 0.5) -> torch.Tensor:
    """Mancha gaussiana fija en el centro de la imagen (coordenadas en [-1, 1]), valores en [0, 1]."""
    axis = torch.linspace(-1.0, 1.0, side, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))


def load_model(run: str, arch: str, device: torch.device) -> torch.nn.Module:
    model = build_model(arch, "full_dlr", pretrained=False)
    model.load_state_dict(torch.load(MODELS_DIR / "vision" / f"{run}.pt", map_location=device, weights_only=True))
    return model.to(device).eval()


def analyze(run: str, arch: str, frame: pd.DataFrame, dataset: str, device: torch.device):
    model = load_model(run, arch, device)
    data = LeafDataset(frame, eval_transform(IMG_SIZE))
    rows, panels = [], []
    for i in range(len(data)):
        x, label = data[i]
        x = x.unsqueeze(0).to(device)
        with GradCAM(model, target_layer(model, arch)) as g:
            cam, logits, idx = g(x)
        pred = int(idx[0])
        di = deletion_insertion(model, x, cam[0], pred, grid=16, n_steps=32, n_random=3)
        # control: un mapa fijo centrado (las hojas de BRACOL están centradas, así que una mancha central ya
        # supera a un orden aleatorio sin saber nada de lesiones); el Grad-CAM debe superarlo para aportar algo
        center = deletion_insertion(model, x, center_prior(x.device), pred, grid=16, n_steps=32, n_random=1)
        di["centro_deletion_auc"], di["centro_insertion_auc"] = center["deletion_auc"], center["insertion_auc"]
        # cuDNN falló de forma intermitente (STREAM_MISMATCH) en el backward del modelo con pesos reiniciados;
        # esta parte se hace sin cuDNN (más lenta, pero son pocas pasadas) y con un reintento
        for attempt in (0, 1):
            try:
                with torch.backends.cudnn.flags(enabled=False):
                    cascade = cascading_randomization(model, arch, x, pred)
                break
            except RuntimeError as exc:
                if attempt == 1:
                    raise
                print(f"   reintento por error de GPU: {str(exc)[:80]}", flush=True)
                torch.cuda.synchronize()
        image = denormalize(x)
        row = {"modelo": run, "dataset": dataset, "real": label, "predicho": pred, **di,
               **{f"spearman{name}": rho for name, rho in cascade[1:]}}
        if dataset == "bracol":
            row.update({f"fondo_{k}": v for k, v in background_attention(image, cam[0].cpu().numpy()).items()})
        rows.append(row)
        panels.append((image, cam[0].cpu().numpy(), label, pred))
        del cascade, di, center
        if device.type == "cuda":      # la copia del modelo de la aleatorización y los lotes de deletion/insertion
            torch.cuda.empty_cache()  # fragmentaban los 4 GB de la GPU y provocaron un error de memoria
    return rows, panels


def figure(panels: Dict[str, list], run: str) -> None:
    n = max(len(p) for p in panels.values())
    cols = min(n, 8)
    fig, axes = plt.subplots(4, cols, figsize=(2.0 * cols, 8.4))
    for block, (dataset, items) in enumerate(panels.items()):
        step = max(1, len(items) // cols)
        for c, (image, cam, label, pred) in enumerate(items[::step][:cols]):
            for r, show in enumerate((image, None)):
                ax = axes[block * 2 + r, c]
                ax.imshow(image)
                if show is None:
                    ax.imshow(cam, cmap="jet", alpha=0.45)
                ax.axis("off")
                if r == 0:
                    ax.set_title(f"{dataset}\nreal {SHORT[label]} → {SHORT[pred]}", fontsize=7,
                                 color="green" if label == pred else "red")
    fig.tight_layout()
    path = ROOT / "docs" / "figures" / f"gradcam_ejemplos_{run}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summarize(table: pd.DataFrame) -> Dict:
    out = {}
    for (run, dataset), g in table.groupby(["modelo", "dataset"]):
        entry = {"imagenes": len(g),
                 "deletion_auc": float(g["deletion_auc"].mean()),
                 "deletion_auc_aleatorio": float(g["deletion_auc_aleatorio"].mean()),
                 "insertion_auc": float(g["insertion_auc"].mean()),
                 "insertion_auc_aleatorio": float(g["insertion_auc_aleatorio"].mean()),
                 "frac_deletion_mejor_que_azar": float((g["deletion_auc"] < g["deletion_auc_aleatorio"]).mean()),
                 "frac_insertion_mejor_que_azar": float((g["insertion_auc"] > g["insertion_auc_aleatorio"]).mean()),
                 "centro_deletion_auc": float(g["centro_deletion_auc"].mean()),
                 "centro_insertion_auc": float(g["centro_insertion_auc"].mean()),
                 "frac_deletion_mejor_que_centro": float((g["deletion_auc"] < g["centro_deletion_auc"]).mean()),
                 "frac_insertion_mejor_que_centro": float((g["insertion_auc"] > g["centro_insertion_auc"]).mean()),
                 "spearman_por_etapa": {c.replace("spearman", ""): float(g[c].mean())
                                        for c in g.columns if c.startswith("spearman") and g[c].notna().any()}}
        if "fondo_cociente" in g and g["fondo_cociente"].notna().any():
            entry["fondo_cociente_medio"] = float(g["fondo_cociente"].mean())
            entry["frac_prefiere_fondo"] = float((g["fondo_cociente"] > 1).mean())
            entry["area_fondo_media"] = float(g["fondo_area_fondo"].mean())
        out[f"{run}|{dataset}"] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-class", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = pd.read_csv(SPLITS_PATH)
    frames = {ds: sample(splits, ds, args.per_class) for ds in ("bracol", "rocole")}
    parts_dir = EXPERIMENTS_DIR / "vision" / "gradcam_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for run, arch in MODELS:
        panels = {}
        for dataset, frame in frames.items():
            part = parts_dir / f"{run}_{dataset}_n{args.per_class}.csv"
            if part.exists():                       # ya analizado: la corrida se puede retomar tras una caída
                print(f"{run} · {dataset}: ya existe {part.name}, se omite", flush=True)
                continue
            r, p = analyze(run, arch, frame, dataset, device)
            pd.DataFrame(r).round(5).to_csv(part, index=False)   # se guarda al terminar cada bloque
            panels[dataset] = p
            print(f"{run} · {dataset}: {len(frame)} imágenes analizadas", flush=True)
        if len(panels) == 2:
            figure(panels, run)
    table = pd.concat([pd.read_csv(p) for p in sorted(parts_dir.glob(f"*_n{args.per_class}.csv"))], ignore_index=True)
    table.round(5).to_csv(EXPERIMENTS_DIR / "vision" / "gradcam_validation.csv", index=False)
    summary = summarize(table)
    (EXPERIMENTS_DIR / "vision" / "gradcam_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for key, v in summary.items():
        print(f"\n== {key} ({v['imagenes']} imágenes)")
        print(f"  deletion  AUC {v['deletion_auc']:.3f} (azar {v['deletion_auc_aleatorio']:.3f}; mejor que azar en "
              f"{v['frac_deletion_mejor_que_azar']:.0%})")
        print(f"  insertion AUC {v['insertion_auc']:.3f} (azar {v['insertion_auc_aleatorio']:.3f}; mejor que azar en "
              f"{v['frac_insertion_mejor_que_azar']:.0%})")
        print("  Spearman vs. original al reiniciar:", {k: round(x, 2) for k, x in v["spearman_por_etapa"].items()})
        if "fondo_cociente_medio" in v:
            print(f"  fondo (BRACOL): cociente medio {v['fondo_cociente_medio']:.2f}, prefiere el fondo en "
                  f"{v['frac_prefiere_fondo']:.0%}; el fondo ocupa {v['area_fondo_media']:.0%} de la imagen")


if __name__ == "__main__":
    main()

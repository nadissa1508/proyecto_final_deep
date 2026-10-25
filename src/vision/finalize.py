"""Calibra, compara candidatos finales y exporta el artefacto de visión.

Candidatos (todos con el mismo split; la selección usa SOLO la validación):

- A: ResNet-18 combinado (BRACOL + RoCoLe), semilla 42. Es la corrida original; no se elige semilla.
- B: ensamble de ResNet-18 solo-BRACOL y ResNet-18 solo-RoCoLe (decisión D3 de PLAN.md).
- C: ensamble de las 4 semillas de ResNet-18 combinado.
- D: ensamble de 4 ResNet-18 y 4 EfficientNet-B0 combinados.
- E: EfficientNet-B0 combinado, semilla 42 (comparación de backbone sin ensamble).

Calibración: a cada miembro se le ajusta una temperatura T sobre la validación **conjunta** de BRACOL y
RoCoLe (en uso real no se sabe de qué dominio viene una foto). La probabilidad del candidato es el
promedio de las de sus miembros calibrados. Regla de selección: menor pérdida logarítmica (NLL) calibrada
en validación; con un solo parámetro por miembro la ventaja que eso da a los ensambles es pequeña, pero
existe y se menciona al reportar.

Se reporta en test (BRACOL y RoCoLe por separado): F1 macro, ECE sin calibrar y calibrado, NLL, recall de
roya severa y detección de roya (sensibilidad y especificidad de leve/severa frente a no roya).

Escribe `experiments/vision/final_comparison.csv`, `experiments/vision/confusion_hoja.json` (matrices de
confusión por hoja del candidato elegido: P(O|L) en cuentas) y `models/vision/artifact.json` (lo lee
`src/vision/infer.py`).

Uso:
    python -m src.vision.finalize
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.common.paths import EXPERIMENTS_DIR, MODELS_DIR
from src.vision.calibration import calibrate, expected_calibration_error, fit_temperature
from src.vision.datasets import CLASSES
from src.vision.models import build_model
from src.vision.splits import SPLITS_PATH
from src.vision.summarize import rust_detection
from src.vision.train import LeafDataset, metrics, predict, select
from src.vision.transforms import eval_transform

IMG_SIZE = 256
SEEDS = ["", "_s1", "_s2", "_s3"]
ARCH = {"resnet18": "resnet18", "efficientnet_b0": "efficientnet_b0"}

CANDIDATES: Dict[str, List[Tuple[str, str]]] = {
    "A_resnet18_combinado_s42": [("resnet18_dlr_both", "resnet18")],
    "B_ensamble_bracol_rocole": [("resnet18_dlr_bracol", "resnet18"), ("resnet18_dlr_rocole", "resnet18")],
    "C_ensamble_4_semillas_resnet18": [(f"resnet18_dlr_both{s}", "resnet18") for s in SEEDS],
    "D_ensamble_resnet18_efficientnet": [(f"resnet18_dlr_both{s}", "resnet18") for s in SEEDS]
                                        + [(f"efficientnet_b0_dlr_both{s}", "efficientnet_b0") for s in SEEDS],
    "E_efficientnet_b0_combinado_s42": [("efficientnet_b0_dlr_both", "efficientnet_b0")],
}


def _loader(frame: pd.DataFrame):
    from torch.utils.data import DataLoader

    return DataLoader(LeafDataset(frame, eval_transform(IMG_SIZE)), batch_size=64, shuffle=False, num_workers=0)


def collect_logits(run: str, arch: str, frames: Dict[str, pd.DataFrame], device: torch.device) -> Dict[str, torch.Tensor]:
    model = build_model(arch, "full_dlr", pretrained=False)
    model.load_state_dict(torch.load(MODELS_DIR / "vision" / f"{run}.pt", map_location=device, weights_only=True))
    model.to(device)
    out = {}
    for name, frame in frames.items():
        logits, _ = predict(model, _loader(frame), device, amp=device.type == "cuda")
        out[name] = logits
    return out


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())


def evaluate_candidate(members, logits: Dict[str, Dict[str, torch.Tensor]], labels: Dict[str, np.ndarray]):
    temps, cal, raw = {}, {}, {}
    for run, _ in members:
        temps[run] = fit_temperature(logits[run]["val"], torch.from_numpy(labels["val"]))
    for split in labels:
        cal[split] = torch.stack([calibrate(logits[r][split], temps[r]) for r, _ in members]).mean(0).numpy()
        raw[split] = torch.stack([torch.softmax(logits[r][split].double(), 1) for r, _ in members]).mean(0).numpy()
    row = {"miembros": len(members), "val_nll_calibrado": nll(cal["val"], labels["val"]),
           "val_f1": metrics(labels["val"], cal["val"].argmax(1))["f1_macro"],
           "temperaturas": {r: round(t, 3) for r, t in temps.items()}}
    confusion = {}
    for ds in ("bracol", "rocole"):
        y, p = labels[f"test_{ds}"], cal[f"test_{ds}"]
        m = metrics(y, p.argmax(1))
        sens, spec = rust_detection(m["matriz_confusion"])
        row.update({f"{ds}_f1": m["f1_macro"], f"{ds}_recall_severa": m["recall_por_clase"]["roya_severa"],
                    f"{ds}_sens_roya": sens, f"{ds}_esp_roya": spec, f"{ds}_nll": nll(p, y),
                    f"{ds}_ece_sin_calibrar": expected_calibration_error(raw[f"test_{ds}"], y),
                    f"{ds}_ece_calibrado": expected_calibration_error(p, y)})
        confusion[ds] = m["matriz_confusion"]
    return row, temps, confusion


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = pd.read_csv(SPLITS_PATH)
    frames = {"val": select(splits, ["bracol", "rocole"], "val"),
              "test_bracol": select(splits, ["bracol"], "test"), "test_rocole": select(splits, ["rocole"], "test")}
    labels = {k: v["label"].astype(int).to_numpy() for k, v in frames.items()}
    runs = sorted({m for members in CANDIDATES.values() for m in members})
    logits = {}
    for run, arch in runs:
        logits[run] = collect_logits(run, arch, frames, device)
        print(f"logits listos: {run}", flush=True)

    rows, details = [], {}
    for name, members in CANDIDATES.items():
        row, temps, confusion = evaluate_candidate(members, logits, labels)
        details[name] = (members, temps, confusion)
        rows.append({"candidato": name, **{k: v for k, v in row.items() if k != "temperaturas"}})
        print(f"{name}: T = {row['temperaturas']}", flush=True)
    table = pd.DataFrame(rows)
    out_dir = EXPERIMENTS_DIR / "vision"
    table.round(4).to_csv(out_dir / "final_comparison.csv", index=False)

    chosen = table.sort_values("val_nll_calibrado").iloc[0]["candidato"]
    members, temps, confusion = details[chosen]
    artifact = {"classes": list(CLASSES), "img_size": IMG_SIZE, "candidato": chosen,
                "models": [{"name": r, "arch": a, "weights": f"{r}.pt", "temperature": round(temps[r], 4)}
                           for r, a in members]}
    (MODELS_DIR / "vision" / "artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    (out_dir / "confusion_hoja.json").write_text(json.dumps({
        "candidato": chosen, "clases": list(CLASSES), "filas": "clase real", "columnas": "clase predicha",
        "test_bracol": confusion["bracol"], "test_rocole": confusion["rocole"]}, indent=2), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\nSelección en validación (menor NLL calibrado):")
    print(table[["candidato", "miembros", "val_nll_calibrado", "val_f1"]].round(4).to_string(index=False))
    for ds in ("bracol", "rocole"):
        print(f"\nTest {ds.upper()}:")
        cols = ["candidato"] + [f"{ds}_{c}" for c in ("f1", "recall_severa", "sens_roya", "esp_roya", "nll",
                                                      "ece_sin_calibrar", "ece_calibrado")]
        print(table[cols].round(3).to_string(index=False))
    print(f"\nElegido por validación: {chosen}")


if __name__ == "__main__":
    main()

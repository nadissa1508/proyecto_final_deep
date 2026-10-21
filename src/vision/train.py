"""Entrenamiento y evaluación del clasificador de hojas de café.

Uso:
    python -m src.vision.train --config configs/vision_resnet18_fe.yaml
    python -m src.vision.train --config configs/vision_resnet18_fe.yaml --smoke

Cada corrida crea `experiments/vision/<fecha>_<nombre>/` con `config.yaml`, `metrics.csv`
(una fila por época) y `results.json` (métricas en val y en el test de BRACOL y de RoCoLe:
exactitud, F1 macro, recall por clase, matriz de confusión). Los pesos del mejor modelo
(por F1 macro en validación) van a `models/vision/<nombre>.pt`, fuera de git.

El test de BRACOL (arábica) es la evaluación principal (PLAN.md, D3). Se evalúan los dos
tests siempre, aunque el modelo no se haya entrenado con ese dataset: esa es la matriz de
evaluación cruzada.

`--smoke` usa pocas imágenes y 2 épocas: sirve para comprobar que la pérdida baja antes de
lanzar una corrida completa.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.common.paths import EXPERIMENTS_DIR, MODELS_DIR
from src.common.seed import set_seed
from src.vision.datasets import CLASSES
from src.vision.models import build_model, count_params, param_groups, set_train_mode
from src.vision.preprocess import cache_path
from src.vision.splits import SPLITS_PATH
from src.vision.transforms import eval_transform, train_transform

LABELS = list(range(len(CLASSES)))


class LeafDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform) -> None:
        self.paths = [cache_path(d, p) for d, p in zip(frame["dataset"], frame["path"])]
        self.labels = frame["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            image = self.transform(im.convert("RGB"))
        return image, self.labels[i]


def select(splits: pd.DataFrame, datasets: List[str], role: str) -> pd.DataFrame:
    return splits[splits["dataset"].isin(datasets) & (splits["split"] == role)]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)), 4),
        "recall_por_clase": {c: round(float(r), 4) for c, r in zip(
            CLASSES, recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0))},
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def loader_for(frame: pd.DataFrame, transform, cfg: dict, shuffle: bool) -> DataLoader:
    workers = cfg["num_workers"]
    return DataLoader(LeafDataset(frame, transform), batch_size=cfg["batch_size"], shuffle=shuffle,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                      drop_last=shuffle)


def run(cfg: dict, smoke: bool = False) -> Path:
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["amp"]) and device.type == "cuda"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = EXPERIMENTS_DIR / "vision" / f"{stamp}_{cfg['name']}{'_smoke' if smoke else ''}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    splits = pd.read_csv(SPLITS_PATH)
    train_sets = ["bracol", "rocole"] if cfg["train_on"] == "both" else [cfg["train_on"]]
    train_df, val_df = select(splits, train_sets, "train"), select(splits, train_sets, "val")
    if smoke:
        train_df = train_df.groupby("label", group_keys=False).head(25)
        val_df = val_df.groupby("label", group_keys=False).head(10)
        cfg = {**cfg, "epochs": 2}

    model = build_model(cfg["model"], cfg["strategy"], pretrained=cfg["strategy"] != "scratch").to(device)
    params = count_params(model)
    groups = param_groups(model, cfg["model"], cfg["strategy"], cfg["lr"],
                          cfg.get("lr_block", cfg["lr"] / 10), cfg.get("lrs_dlr", []))
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler(enabled=amp)

    counts = train_df["label"].value_counts().reindex(LABELS).fillna(1).to_numpy()
    weight = torch.tensor(len(train_df) / (len(LABELS) * counts), dtype=torch.float32) if cfg["class_weights"] else None
    criterion = nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None)

    train_loader = loader_for(train_df, train_transform(cfg["img_size"]), cfg, shuffle=True)
    val_loader = loader_for(val_df, eval_transform(cfg["img_size"]), cfg, shuffle=False)
    print(f"[{cfg['name']}] {device} · train {len(train_df)} · val {len(val_df)} · params {params}", flush=True)

    best_f1, best_epoch, stale, history = -1.0, 0, 0, []
    weights_path = MODELS_DIR / "vision" / f"{cfg['name']}{'_smoke' if smoke else ''}.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        set_train_mode(model)
        losses = []
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()
        logits, labels = predict(model, val_loader, device, amp)
        val = metrics(labels.numpy(), logits.argmax(1).numpy())
        val_loss = float(nn.functional.cross_entropy(logits, labels, weight=weight).item()) if weight is not None \
            else float(nn.functional.cross_entropy(logits, labels).item())
        row = {"epoca": epoch, "loss_train": round(float(np.mean(losses)), 4), "loss_val": round(val_loss, 4),
               "f1_macro_val": val["f1_macro"], "acc_val": val["accuracy"]}
        history.append(row)
        print(row, flush=True)
        if val["f1_macro"] > best_f1:
            best_f1, best_epoch, stale = val["f1_macro"], epoch, 0
            torch.save(model.state_dict(), weights_path)
        else:
            stale += 1
            if stale >= cfg["patience"]:
                print(f"Early stopping en la época {epoch}", flush=True)
                break
    pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    results = {"nombre": cfg["name"], "configuracion": cfg, "parametros": params, "mejor_epoca": best_epoch,
               "f1_macro_val": best_f1, "segundos": round(time.time() - started, 1),
               "loss_train_baja": bool(history[-1]["loss_train"] < history[0]["loss_train"]), "test": {}}
    for dataset in ("bracol", "rocole"):
        test_df = select(splits, [dataset], "test")
        if smoke:
            test_df = test_df.groupby("label", group_keys=False).head(10)
        logits, labels = predict(model, loader_for(test_df, eval_transform(cfg["img_size"]), cfg, False), device, amp)
        results["test"][dataset] = metrics(labels.numpy(), logits.argmax(1).numpy())
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    bt, rt = results["test"]["bracol"], results["test"]["rocole"]
    print(f"[{cfg['name']}] mejor época {best_epoch} · F1 val {best_f1:.3f} · "
          f"test BRACOL F1 {bt['f1_macro']:.3f} (recall {bt['recall_por_clase']}) · test RoCoLe F1 {rt['f1_macro']:.3f}",
          flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()

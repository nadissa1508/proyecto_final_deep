"""Split train/val/test de las imágenes, sin fuga de datos, y baselines.

Parte del manifiesto de `src.vision.datasets` (solo imágenes no excluidas) y asigna
cada **grupo** completo a un único conjunto:

- RoCoLe: grupo = planta (4 fotos por planta) fusionada con sus duplicados.
- BRACOL: grupo = hoja (una foto por hoja).

El split de cada dataset se hace con `StratifiedGroupKFold` (20 pliegues: 14 train, 3 val,
3 test ≈ 70/15/15), que mantiene la proporción de clases y nunca separa un grupo.
El **test de BRACOL (arábica) es la evaluación principal** del proyecto (PLAN.md, D3).

Baselines (se calculan antes de entrenar nada): clase mayoritaria del train y azar
uniforme, medidos en val y test de cada dataset.

Uso:
    python -m src.vision.splits
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src.common.paths import PROCESSED_DIR
from src.vision.datasets import CLASSES, MANIFEST_PATH, build_manifest

SPLITS_PATH = PROCESSED_DIR / "vision" / "splits.csv"
N_FOLDS = 20
FOLD_ROLES = ("train",) * 14 + ("val",) * 3 + ("test",) * 3
SEED = 42


def make_splits(manifest: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    usable = manifest[manifest["excluded"] == ""].copy()
    usable["split"] = ""
    for dataset, part in usable.groupby("dataset"):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        roles = np.random.RandomState(seed).permutation(FOLD_ROLES)
        for fold, (_, idx) in enumerate(sgkf.split(part, part["label"], part["group"])):
            usable.loc[part.index[idx], "split"] = roles[fold]
    return usable


def check_no_leakage(splits: pd.DataFrame) -> None:
    for column in ("group", "sha256"):
        per = splits.groupby(column)["split"].nunique()
        bad = per[per > 1]
        if len(bad):
            raise AssertionError(f"Fuga: {len(bad)} valores de '{column}' están en más de un conjunto")


def baselines(splits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in ("bracol", "rocole", "ambos"):
        part = splits if dataset == "ambos" else splits[splits["dataset"] == dataset]
        train = part[part["split"] == "train"]
        majority = int(train["label"].mode().iloc[0])
        rng = np.random.RandomState(SEED)
        for split in ("val", "test"):
            y = part.loc[part["split"] == split, "label"].astype(int).to_numpy()
            for name, pred in (("clase mayoritaria", np.full(len(y), majority)),
                               ("azar uniforme", rng.randint(0, len(CLASSES), len(y)))):
                rows.append({
                    "datos": dataset, "conjunto": split, "baseline": name, "n": len(y),
                    "accuracy": round(accuracy_score(y, pred), 3),
                    "f1_macro": round(f1_score(y, pred, average="macro", zero_division=0), 3),
                })
    return pd.DataFrame(rows)


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH) if MANIFEST_PATH.exists() else build_manifest()
    manifest["excluded"] = manifest["excluded"].fillna("")
    splits = make_splits(manifest)
    check_no_leakage(splits)
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(SPLITS_PATH, index=False)
    print(f"Splits guardados en {SPLITS_PATH} ({len(splits)} imágenes, sin fuga por grupo ni SHA-256)\n")
    table = pd.crosstab([splits["dataset"], splits["split"]], splits["label"].map(dict(enumerate(CLASSES))))
    print(table[[c for c in CLASSES if c in table.columns]].to_string(), "\n")
    print(baselines(splits).to_string(index=False))


if __name__ == "__main__":
    main()

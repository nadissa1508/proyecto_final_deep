"""Preprocesamiento y particionado para los modelos del módulo ML.

Implementaciones propias en numpy/pandas: escalado Min-Max, codificación de
etiquetas, división estratificada train/test y generador de k-fold
estratificado. Se evita scikit-learn en este archivo para que sea
reutilizable por `champion.py` (implementación desde cero del modelo
ganador).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd


CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CROP_LABEL = "label"


@dataclass
class MinMaxScaler:
    min_: np.ndarray = None
    range_: np.ndarray = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self.min_ = X.min(axis=0)
        self.range_ = X.max(axis=0) - self.min_
        self.range_ = np.where(self.range_ == 0, 1.0, self.range_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


@dataclass
class LabelEncoder:
    classes_: np.ndarray = None

    def fit(self, y: Sequence) -> "LabelEncoder":
        self.classes_ = np.array(sorted(set(y)))
        return self

    def transform(self, y: Sequence) -> np.ndarray:
        idx = {c: i for i, c in enumerate(self.classes_)}
        return np.array([idx[v] for v in y], dtype=int)

    def fit_transform(self, y: Sequence) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y_idx: np.ndarray) -> np.ndarray:
        return self.classes_[y_idx]


def load_crop_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Carga crop_recommendation.csv y devuelve (X, y_index, classes)."""
    df = pd.read_csv(path)
    X = df[CROP_FEATURES].to_numpy(dtype=float)
    encoder = LabelEncoder().fit(df[CROP_LABEL].tolist())
    y = encoder.transform(df[CROP_LABEL].tolist())
    return X, y, list(encoder.classes_)


def stratified_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split estratificado por etiqueta. Devuelve X_tr, X_te, y_tr, y_te."""
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = int(round(len(cls_idx) * test_size))
        test_idx.extend(cls_idx[:n_test])
        train_idx.extend(cls_idx[n_test:])
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def stratified_kfold(
    y: np.ndarray, n_splits: int = 5, seed: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Genera (train_idx, val_idx) para cada fold, preservando proporciones."""
    rng = np.random.default_rng(seed)
    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        for i, idx in enumerate(cls_idx):
            folds[i % n_splits].append(int(idx))
    fold_arrays = [np.array(f) for f in folds]
    all_idx = np.arange(len(y))
    for k in range(n_splits):
        val = fold_arrays[k]
        train = np.setdiff1d(all_idx, val)
        yield train, val

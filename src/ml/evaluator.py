"""Métricas de clasificación multiclase desde cero.

Funciones independientes del modelo: cualquier estimador con interfaz
`.fit(X, y)` y `.predict(X)` puede pasar por `evaluate`. Se usan tanto
en el notebook comparativo (modelos sklearn) como para validar la
implementación propia de `champion.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np


@dataclass
class Metrics:
    accuracy: float
    f1_macro: float
    precision_per_class: np.ndarray
    recall_per_class: np.ndarray
    f1_per_class: np.ndarray
    confusion: np.ndarray
    fit_time_sec: float = 0.0
    predict_time_sec: float = 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1(cm: np.ndarray) -> tuple:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return precision, recall, f1


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def evaluate(
    model: Any, X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, n_classes: int,
) -> Metrics:
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time = time.perf_counter() - t0

    cm = confusion_matrix(y_test, y_pred, n_classes)
    prec, rec, f1 = precision_recall_f1(cm)
    return Metrics(
        accuracy=accuracy(y_test, y_pred),
        f1_macro=float(f1.mean()),
        precision_per_class=prec,
        recall_per_class=rec,
        f1_per_class=f1,
        confusion=cm,
        fit_time_sec=fit_time,
        predict_time_sec=pred_time,
    )


def kfold_evaluate(
    model_factory: Callable[[], Any],
    X: np.ndarray, y: np.ndarray, n_classes: int,
    fold_iter,
) -> Dict[str, float]:
    accs: List[float] = []
    f1s: List[float] = []
    for train_idx, val_idx in fold_iter:
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        cm = confusion_matrix(y[val_idx], y_pred, n_classes)
        _, _, f1 = precision_recall_f1(cm)
        accs.append(accuracy(y[val_idx], y_pred))
        f1s.append(float(f1.mean()))
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
    }

"""Implementación desde cero de Gaussian Naive Bayes — modelo ganador del notebook 04.

Asume independencia condicional de las features dado el cultivo y verosimilitud
gaussiana por feature.  Para clase `c` y vector de features `x`:

    log P(c | x) = log P(c) + sum_j log N(x_j ; mu_{c,j}, sigma2_{c,j}) + const

El ajuste estima `mu` y `sigma2` por clase con suavizado de varianza estilo
`sklearn` (`var_smoothing * max(var)` se suma a las varianzas) para evitar
divisiones por cero en features con valor constante en una clase.

La predicción de probabilidad usa la log-sum-exp trick para mantener estabilidad
numérica con 22 clases.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class Champion:
    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing
        self.classes_: Optional[np.ndarray] = None
        self.class_prior_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Champion":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)

        epsilon = self.var_smoothing * X.var(axis=0).max()
        for i, c in enumerate(self.classes_):
            mask = y == c
            Xc = X[mask]
            self.class_prior_[i] = mask.sum() / len(y)
            self.theta_[i] = Xc.mean(axis=0)
            self.var_[i] = Xc.var(axis=0) + epsilon
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        jll = np.empty((n_samples, n_classes))
        for i in range(n_classes):
            log_prior = np.log(self.class_prior_[i])
            log_norm = -0.5 * np.sum(np.log(2 * np.pi * self.var_[i]))
            sq_diff = ((X - self.theta_[i]) ** 2) / self.var_[i]
            jll[:, i] = log_prior + log_norm - 0.5 * sq_diff.sum(axis=1)
        return jll

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        jll = self._joint_log_likelihood(X)
        log_norm = jll.max(axis=1, keepdims=True)
        exp = np.exp(jll - log_norm)
        return exp / exp.sum(axis=1, keepdims=True)

"""
Z-score standardisation, written out by hand so the inverse is obvious.

Why standardise? The SOM uses Euclidean distance. Without scaling, Log_Traded_Value
(~20) would dominate Correlation (~0.5) simply because its numbers are bigger.
After z-scoring every feature has mean 0 and std 1, so each one gets an equal vote.

The scaler is fitted on the CLUSTERING period features only and kept, so the UI can
turn neuron weights (in z-units) back into real units (%, x, Rs).
"""

from __future__ import annotations

import numpy as np


class ZScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ZScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)                 # population std (ddof=0), like sklearn
        self.std_[self.std_ == 0] = 1.0           # a constant column would divide by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """z -> real units:  x = z * std + mean."""
        return Z * self.std_ + self.mean_

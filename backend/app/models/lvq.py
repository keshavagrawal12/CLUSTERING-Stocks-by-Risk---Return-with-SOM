"""
Learning Vector Quantization (LVQ1), written from scratch in NumPy.
Syllabus: Unit 3 - "Learning Vector Quantization".

SOM vs LVQ - both are COMPETITIVE networks (a winner-takes-all layer of prototype neurons):
  * SOM  = UNSUPERVISED competitive learning: no labels, neighbours learn too, the result
           is a topology-preserving map.
  * LVQ  = SUPERVISED competitive learning: every prototype carries a class label (here a
           sector), there is no neighbourhood, and the label decides the DIRECTION of the
           update.

LVQ1 rule for one training sample (x, y):
  1. find the winning prototype w_c (closest to x)          - competition, as in the SOM
  2. if class(w_c) == y :  w_c <- w_c + eta(t) * (x - w_c)   (pull TOWARDS x)
     else               :  w_c <- w_c - eta(t) * (x - w_c)   (push AWAY from x)

In this project LVQ asks: "Given only a stock's behaviour, can a competitive network
guess its official sector?" If it can't beat the majority-class baseline, behaviour and
sector labels are only loosely related - a second angle on RQ1.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold

from app import config


def merge_small_classes(labels: np.ndarray, min_size: int = config.LVQ_MIN_CLASS_SIZE) -> np.ndarray:
    """
    Sectors with fewer than `min_size` stocks are merged into "Other" (LVQ only).
    Why: a class with 1-2 stocks cannot appear in every CV fold, and one prototype fitted
    to one stock is not a meaningful "sector prototype".
    """
    labels = np.asarray(labels).astype(object)
    values, counts = np.unique(labels, return_counts=True)
    small = set(values[counts < min_size])
    return np.array(["Other" if l in small else l for l in labels], dtype=object)


class LVQ1:
    def __init__(self, eta0: float = config.LVQ_ETA0, epochs: int = config.LVQ_EPOCHS,
                 prototypes_per_class: int = config.LVQ_PROTOTYPES_PER_CLASS,
                 init_noise: float = config.LVQ_INIT_NOISE, seed: int = config.LVQ_SEED) -> None:
        self.eta0, self.epochs = eta0, epochs
        self.ppc, self.init_noise = prototypes_per_class, init_noise
        self.rng = np.random.default_rng(seed)
        self.prototypes: np.ndarray | None = None      # (P, dim)
        self.proto_labels: np.ndarray | None = None    # (P,)

    def _init_prototypes(self, X: np.ndarray, y: np.ndarray) -> None:
        """Start each prototype at its class mean plus a little noise (so duplicates differ)."""
        protos, labels = [], []
        for c in np.unique(y):
            mean = X[y == c].mean(axis=0)
            for _ in range(self.ppc):
                protos.append(mean + self.rng.normal(0, self.init_noise, size=X.shape[1]))
                labels.append(c)
        self.prototypes = np.array(protos, dtype=float)
        self.proto_labels = np.array(labels, dtype=object)

    def winner(self, x: np.ndarray) -> int:
        """Competitive step: index of the closest prototype."""
        return int(np.argmin(np.linalg.norm(self.prototypes - x, axis=1)))

    def step(self, x: np.ndarray, y, eta: float) -> int:
        """One LVQ1 update. Returns the winner index."""
        c = self.winner(x)
        if self.proto_labels[c] == y:
            self.prototypes[c] += eta * (x - self.prototypes[c])   # correct -> move towards
        else:
            self.prototypes[c] -= eta * (x - self.prototypes[c])   # wrong   -> move away
        return c

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LVQ1":
        self._init_prototypes(X, y)
        total = self.epochs * len(X)
        t = 0
        for _ in range(self.epochs):
            for i in self.rng.permutation(len(X)):
                # Linear decay of the learning rate to 0 over training (classic LVQ1 choice).
                eta = self.eta0 * (1 - t / total)
                self.step(X[i], y[i], eta)
                t += 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.proto_labels[self.winner(x)] for x in X], dtype=object)


@dataclass
class LVQEvaluation:
    accuracy: float
    accuracy_std: float
    fold_accuracies: list[float]
    baseline_accuracy: float
    baseline_class: str
    classes: list[str]
    confusion: list[list[int]]          # rows = true class, cols = predicted class
    merged_into_other: list[str]
    n_samples: int
    folds: int


def evaluate_lvq(X: np.ndarray, sectors: np.ndarray, folds: int = config.LVQ_FOLDS,
                 seed: int = config.LVQ_SEED, **lvq_kwargs) -> LVQEvaluation:
    """
    Stratified k-fold cross-validation (keeps each sector's share similar in every fold -
    important with only ~100 stocks). Baseline = always predict the most common sector
    of the TRAINING fold.
    """
    original = np.asarray(sectors).astype(object)
    y = merge_small_classes(original)
    merged = sorted(set(original[y == "Other"]) - {"Other"})
    classes = sorted(np.unique(y).tolist())
    idx = {c: i for i, c in enumerate(classes)}

    # After merging, every class has >= LVQ_MIN_CLASS_SIZE stocks, so every class is always
    # present in each TRAINING split. With 5 folds a 3-stock sector simply can't appear in
    # every TEST fold - sklearn warns about that, which is expected here.
    k = max(2, min(folds, len(y) // 2))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    conf = np.zeros((len(classes), len(classes)), dtype=int)
    accs, base_hits = [], 0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class")
        splits = list(skf.split(X, y))
    for train, test in splits:
        model = LVQ1(seed=seed, **lvq_kwargs).fit(X[train], y[train])
        pred = model.predict(X[test])
        accs.append(float(np.mean(pred == y[test])))
        for t_, p_ in zip(y[test], pred):
            conf[idx[t_], idx[p_]] += 1
        vals, cnts = np.unique(y[train], return_counts=True)
        majority = vals[np.argmax(cnts)]
        base_hits += int(np.sum(y[test] == majority))

    vals, cnts = np.unique(y, return_counts=True)
    return LVQEvaluation(
        accuracy=float(np.trace(conf) / conf.sum()),
        accuracy_std=float(np.std(accs)),
        fold_accuracies=accs,
        baseline_accuracy=float(base_hits / len(y)),
        baseline_class=str(vals[np.argmax(cnts)]),
        classes=classes,
        confusion=conf.tolist(),
        merged_into_other=merged,
        n_samples=int(len(y)),
        folds=k,
    )

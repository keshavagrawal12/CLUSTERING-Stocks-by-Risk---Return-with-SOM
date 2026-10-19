"""
Turning SOM neurons into clusters, plus the plain K-means baseline.

Why a second step? A 7x7 SOM has 49 neurons - that's 49 tiny "micro-clusters", far too
many to interpret. Neighbouring neurons hold similar prototypes, so we group the 49
trained WEIGHT VECTORS with K-means. Each stock then inherits the cluster of its BMU.
k is chosen by the silhouette score (how much closer each neuron is to its own cluster
than to the next-nearest cluster; ranges -1..1, higher is better).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from app import config


@dataclass
class NeuronClustering:
    k: int
    silhouette: list[dict]              # [{"k": 3, "score": 0.41}, ...]
    neuron_labels: np.ndarray           # (rows, cols) cluster id per neuron
    stock_labels: np.ndarray            # (N,) cluster id per stock (via its BMU)


def cluster_neurons(weights: np.ndarray, bmus: np.ndarray, seed: int = config.SOM_SEED,
                    k_min: int = config.CLUSTER_K_MIN, k_max: int = config.CLUSTER_K_MAX) -> NeuronClustering:
    rows, cols, dim = weights.shape
    W = weights.reshape(-1, dim)
    k_max = min(k_max, len(W) - 1)
    curve, best = [], None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=config.KMEANS_N_INIT, random_state=seed).fit(W)
        score = float(silhouette_score(W, km.labels_))
        curve.append({"k": k, "score": score})
        if best is None or score > best[1]:
            best = (k, score, km.labels_)
    k, _, labels = best
    labels = _relabel_by_size(labels, bmus, cols)
    neuron_labels = labels.reshape(rows, cols)
    stock_labels = np.array([neuron_labels[r, c] for r, c in bmus], dtype=int)
    return NeuronClustering(k=k, silhouette=curve, neuron_labels=neuron_labels, stock_labels=stock_labels)


def _relabel_by_size(labels: np.ndarray, bmus: np.ndarray, cols: int) -> np.ndarray:
    """Renumber clusters so cluster 0 holds the most stocks (stable, readable ids in the UI)."""
    stock_lbl = labels[bmus[:, 0] * cols + bmus[:, 1]]
    uniq = np.unique(labels)
    counts = {u: int(np.sum(stock_lbl == u)) for u in uniq}
    order = sorted(uniq, key=lambda u: (-counts[u], u))
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in labels], dtype=int)


def kmeans_baseline(X: np.ndarray, k: int, seed: int = config.SOM_SEED) -> np.ndarray:
    """Plain K-means directly on the standardised stock features (no neural network)."""
    return KMeans(n_clusters=k, n_init=config.KMEANS_N_INIT, random_state=seed).fit_predict(X)

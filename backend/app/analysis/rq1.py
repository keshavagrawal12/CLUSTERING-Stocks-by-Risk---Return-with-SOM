"""
RQ1 - Do behaviour-based clusters match the official sectors?

Two agreement scores between a clustering and the sector labels:
  * ARI (Adjusted Rand Index): over all PAIRS of stocks, how often do the two groupings
    agree on "same group / different group", corrected for chance. 0 = no better than
    random, 1 = identical groupings.
  * NMI (Normalized Mutual Information): how much knowing a stock's cluster tells you
    about its sector (0 = nothing, 1 = everything).
Both ignore the cluster ids themselves - only the grouping matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def agreement(clusters: np.ndarray, sectors: np.ndarray) -> dict:
    return {
        "ari": float(adjusted_rand_score(sectors, clusters)),
        "nmi": float(normalized_mutual_info_score(sectors, clusters)),
    }


def contingency(clusters: np.ndarray, sectors: np.ndarray) -> dict:
    """Clusters x industries table of stock counts (rows = cluster ids, cols = industries)."""
    tab = pd.crosstab(pd.Series(clusters, name="cluster"), pd.Series(sectors, name="industry"))
    # Order industries by size so the heatmap reads largest-first.
    tab = tab[tab.sum(axis=0).sort_values(ascending=False).index]
    return {
        "clusters": [int(c) for c in tab.index],
        "industries": [str(i) for i in tab.columns],
        "counts": tab.to_numpy().astype(int).tolist(),
    }


def headline(ari: float) -> str:
    """Plain-English reading of an ARI value (thresholds are the usual rough guide)."""
    if ari < 0.05:
        strength = "do not match"
    elif ari < 0.2:
        strength = "only weakly match"
    elif ari < 0.5:
        strength = "partly match"
    else:
        strength = "largely match"
    return f"ARI = {ari:.2f}: behaviour clusters {strength} official sectors"

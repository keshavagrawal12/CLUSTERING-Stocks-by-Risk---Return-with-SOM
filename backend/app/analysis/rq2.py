"""
RQ2 - Which stocks behave differently from the rest of their sector?

For each stock we measure two things on the trained SOM:
  1. Grid distance: how far its BMU is from the centre of mass of the OTHER stocks in its
     sector (leave-one-out, so a misfit doesn't drag the centre towards itself).
     The SOM preserves topology, so far apart on the grid = different behaviour.
  2. Cluster mismatch: is its cluster different from its sector's most common cluster?

Stocks are ranked by grid distance; the explanation compares the stock's features with
its sector's median (in z-units so different features are comparable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app import config

# Plain-English names for the "why" sentence.
FEATURE_WORDS = {
    "Annual_Return": "return",
    "Volatility": "volatility",
    "Beta": "beta",
    "Correlation": "market correlation",
    "Max_Drawdown": "drawdown",          # less negative = smaller drawdown
    "Sharpe_Ratio": "Sharpe ratio",
    "Log_Traded_Value": "traded value",
}


def _explain(z_diff: pd.Series, sector: str, top: int = 2) -> str:
    """e.g. 'Much higher beta and higher volatility than other FMCG stocks'."""
    parts = []
    for feat in z_diff.abs().sort_values(ascending=False).index[:top]:
        d = z_diff[feat]
        if abs(d) < 0.5:
            continue
        if feat == "Max_Drawdown":
            word = "deeper drawdown" if d < 0 else "shallower drawdown"
        else:
            word = ("higher " if d > 0 else "lower ") + FEATURE_WORDS[feat]
        if abs(d) >= 1.5:
            word = "much " + word
        parts.append(word)
    if not parts:
        return f"Close to the {sector} median on every feature; differs only in combination"
    text = " and ".join(parts)
    return f"{text[0].upper()}{text[1:]} than other {sector} stocks"


def find_misfits(tickers: list[str], sectors: np.ndarray, bmus: np.ndarray, clusters: np.ndarray,
                 features: pd.DataFrame, X: np.ndarray, names: dict[str, str]) -> list[dict]:
    """Return every stock (in a sector of >= RQ2_MIN_SECTOR_SIZE) ranked by misfit distance."""
    sectors = np.asarray(sectors)
    z = pd.DataFrame(X, index=tickers, columns=config.FEATURE_COLUMNS)
    out = []
    for s in np.unique(sectors):
        members = np.where(sectors == s)[0]
        if len(members) < config.RQ2_MIN_SECTOR_SIZE:
            continue
        vals, counts = np.unique(clusters[members], return_counts=True)
        modal = int(vals[np.argmax(counts)])
        med_real = features.iloc[members].median()
        med_z = z.iloc[members].median()
        for i in members:
            others = members[members != i]
            centre = bmus[others].mean(axis=0)                 # leave-one-out centre of mass
            dist = float(np.linalg.norm(bmus[i] - centre))
            t = tickers[i]
            out.append({
                "ticker": t,
                "company": names.get(t, t),
                "industry": str(s),
                "sector_size": int(len(members)),
                "bmu": [int(bmus[i][0]), int(bmus[i][1])],
                "sector_centre": [float(centre[0]), float(centre[1])],
                "grid_distance": dist,
                "cluster": int(clusters[i]),
                "sector_modal_cluster": modal,
                "cluster_mismatch": bool(clusters[i] != modal),
                "features": {k: float(v) for k, v in features.iloc[i].items()},
                "sector_median": {k: float(v) for k, v in med_real.items()},
                "z_diff": {k: float(v) for k, v in (z.iloc[i] - med_z).items()},
                "explanation": _explain(z.iloc[i] - med_z, str(s)),
            })
    out.sort(key=lambda r: (-r["grid_distance"], not r["cluster_mismatch"]))
    for rank, r in enumerate(out, 1):
        r["rank"] = rank
    return out

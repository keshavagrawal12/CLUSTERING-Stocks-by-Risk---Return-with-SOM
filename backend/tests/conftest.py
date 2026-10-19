"""
Shared test fixtures. SYNTHETIC DATA LIVES ONLY HERE - the app itself never fabricates data.

`synthetic_csvs` writes three CSVs in the exact format of the Colab notebook:
40 stocks in 5 industries, business days 2022-2025, prices generated from a one-factor
market model  r_stock = beta * r_market + noise,  so each industry has a known beta.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402

INDUSTRIES = {   # industry: (beta, idiosyncratic daily vol, daily drift)
    "Banking": (1.3, 0.012, 0.0006),
    "IT": (0.7, 0.010, 0.0004),
    "FMCG": (0.5, 0.008, 0.0003),
    "Metals": (1.6, 0.020, 0.0002),
    "Pharma": (0.6, 0.014, 0.0005),
}
PER_INDUSTRY = 8


def make_market(seed: int = 0, start: str = "2022-01-03", end: str = "2025-12-31"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    r_m = rng.normal(0.0004, 0.01, len(dates))
    return dates, r_m, rng


def make_frames(seed: int = 0, short_history: tuple[str, ...] = ()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates, r_m, rng = make_market(seed)
    prices = {config.INDEX_TICKER: 15000 * np.cumprod(1 + r_m)}
    volumes = {config.INDEX_TICKER: np.full(len(dates), 1e6)}
    meta = []
    for ind, (beta, vol, drift) in INDUSTRIES.items():
        for j in range(PER_INDUSTRY):
            sym = f"{ind[:3].upper()}{j}"
            t = f"{sym}.NS"
            r = drift + beta * r_m + rng.normal(0, vol, len(dates))
            p = 100 * np.cumprod(1 + r)
            v = rng.lognormal(12 + j * 0.1, 0.3, len(dates))
            if t in short_history:            # listed late -> mostly missing in clustering period
                p[: int(len(dates) * 0.5)] = np.nan
                v[: int(len(dates) * 0.5)] = np.nan
            prices[t], volumes[t] = p, v
            meta.append({"Company Name": f"{ind} Co {j}", "Industry": ind, "Symbol": sym, "Ticker": t})
    pf = pd.DataFrame(prices, index=dates)
    vf = pd.DataFrame(volumes, index=dates)
    pf.index.name = vf.index.name = "Date"
    feats = pd.DataFrame(meta)
    for c in config.FEATURE_COLUMNS:          # placeholder numbers; the loader recomputes them
        feats[c] = 0.0
    return feats, pf, vf


def to_bytes(df: pd.DataFrame, index: bool) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=index)
    return buf.getvalue().encode()


@pytest.fixture
def synthetic_csvs(tmp_path: Path) -> Path:
    feats, pf, vf = make_frames()
    feats.to_csv(tmp_path / config.FEATURES_FILE, index=False)
    pf.to_csv(tmp_path / config.PRICES_FILE)
    vf.to_csv(tmp_path / config.VOLUMES_FILE)
    return tmp_path


@pytest.fixture
def blobs():
    """Three well-separated Gaussian blobs in 7-D, with labels."""
    rng = np.random.default_rng(1)
    centres = np.array([[3, 0, 0, 0, 0, 0, 0], [0, 3, 0, 0, 0, 0, 0], [0, 0, 3, 0, 0, 0, 0]], dtype=float)
    X = np.vstack([c + rng.normal(0, 0.3, (40, 7)) for c in centres])
    y = np.repeat(np.array(["A", "B", "C"], dtype=object), 40)
    return X, y

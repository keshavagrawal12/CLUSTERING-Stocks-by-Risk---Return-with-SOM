"""
Load the three CSVs produced by the Colab notebook and turn them into a `Dataset`.

Flow:
  1. read nifty100_features.csv  -> stock metadata (company, industry, symbol, ticker)
  2. read prices + volumes       -> daily frames indexed by Date
  3. slice prices/volumes to the CLUSTERING period and RE-COMPUTE the 7 features
     (so we know exactly which days went into them - no 2025 data can leak in)
  4. compare our recomputed features with the CSV's columns (sanity check, shown in UI)
  5. z-score the features (scaler fitted here, on clustering-period features only)
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app import config
from app.data.features import compute_features, slice_period
from app.data.scaler import ZScaler

log = logging.getLogger(__name__)

META_COLUMNS = ["Company Name", "Industry", "Symbol", "Ticker"]


class DataError(ValueError):
    """Raised when an uploaded/placed file has the wrong shape. Message is shown to the user."""


@dataclass
class Dataset:
    meta: pd.DataFrame                 # index = ticker; Company Name, Industry, Symbol
    features: pd.DataFrame             # index = ticker; 7 features in real units
    X: np.ndarray                      # standardised features, same row order as `features`
    scaler: ZScaler
    prices: pd.DataFrame               # full price history (both periods) - RQ3 slices it itself
    dropped: list[dict] = field(default_factory=list)
    feature_check: dict = field(default_factory=dict)   # max |ours - csv| per feature
    source: str = "files"

    @property
    def tickers(self) -> list[str]:
        return list(self.features.index)

    @property
    def industries(self) -> np.ndarray:
        return self.meta.loc[self.features.index, "Industry"].to_numpy()


def _read_features_csv(src) -> pd.DataFrame:
    df = pd.read_csv(src)
    missing = [c for c in META_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"nifty100_features.csv is missing columns: {missing}")
    df = df.drop_duplicates(subset="Ticker").set_index("Ticker")
    return df


def _read_daily_csv(src, name: str) -> pd.DataFrame:
    df = pd.read_csv(src)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], utc=False, errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    # yfinance sometimes writes tz-aware timestamps; strip tz so slicing by date works.
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.empty:
        raise DataError(f"{name} has no rows")
    return df


def build_dataset(features_src, prices_src, volumes_src, source: str = "files") -> Dataset:
    """Build a Dataset from three file paths or file-like objects."""
    meta_df = _read_features_csv(features_src)
    prices = _read_daily_csv(prices_src, config.PRICES_FILE)
    volumes = _read_daily_csv(volumes_src, config.VOLUMES_FILE) if volumes_src is not None else None

    if config.INDEX_TICKER not in prices.columns:
        raise DataError(f"{config.PRICES_FILE} must contain the NIFTY 50 column '{config.INDEX_TICKER}'")

    # --- The leakage gate: features only ever see the clustering-period slice. ---
    cl_prices = slice_period(prices, config.CLUSTER_START, config.CLUSTER_END)
    cl_volumes = slice_period(volumes, config.CLUSTER_START, config.CLUSTER_END) if volumes is not None else None
    if cl_prices.empty:
        raise DataError("No price rows fall inside the clustering period "
                        f"{config.CLUSTER_START} .. {config.CLUSTER_END}")

    tickers = [t for t in meta_df.index if t != config.INDEX_TICKER]
    result = compute_features(cl_prices, cl_volumes, tickers)
    feats = result.features
    dropped = list(result.dropped)

    # If volumes were missing for some ticker, fall back to the CSV's own Log_Traded_Value.
    if "Log_Traded_Value" in meta_df.columns:
        na = feats["Log_Traded_Value"].isna()
        feats.loc[na, "Log_Traded_Value"] = meta_df.loc[feats.index[na], "Log_Traded_Value"]

    bad = feats.index[feats.isna().any(axis=1) | ~np.isfinite(feats.to_numpy(dtype=float)).all(axis=1)]
    for t in bad:
        dropped.append({"ticker": t, "reason": "a feature could not be computed (NaN/inf)"})
    feats = feats.drop(index=bad)

    if len(feats) < config.CLUSTER_K_MAX + 1:
        raise DataError(f"Only {len(feats)} usable stocks after cleaning - need at least "
                        f"{config.CLUSTER_K_MAX + 1} to cluster.")

    # Sanity check vs the notebook's numbers (tiny differences = rounding / NaN handling).
    check = {}
    for c in config.FEATURE_COLUMNS:
        if c in meta_df.columns:
            diff = (feats[c] - pd.to_numeric(meta_df.loc[feats.index, c], errors="coerce")).abs()
            check[c] = None if diff.isna().all() else float(diff.max())

    scaler = ZScaler()
    X = scaler.fit_transform(feats.to_numpy(dtype=float))

    meta = meta_df.loc[feats.index, ["Company Name", "Industry", "Symbol"]].copy()
    meta["Industry"] = meta["Industry"].fillna("Unknown").astype(str).str.strip()
    log.info("Loaded %d stocks (%d dropped)", len(feats), len(dropped))
    return Dataset(meta=meta, features=feats, X=X, scaler=scaler, prices=prices,
                   dropped=dropped, feature_check=check, source=source)


def load_from_dir(folder: Path = config.RAW_DATA_DIR) -> Dataset | None:
    """Load from data/raw/ if all three files are there; return None otherwise (empty state)."""
    f, p, v = (folder / config.FEATURES_FILE, folder / config.PRICES_FILE, folder / config.VOLUMES_FILE)
    if not (f.exists() and p.exists() and v.exists()):
        return None
    return build_dataset(f, p, v, source=f"folder: {folder}")


def load_from_bytes(features: bytes, prices: bytes, volumes: bytes) -> Dataset:
    """Used by the upload endpoint."""
    return build_dataset(io.BytesIO(features), io.BytesIO(prices), io.BytesIO(volumes), source="upload")

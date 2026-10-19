"""
Rebuild the three CSVs from scratch with yfinance (mirrors the Colab notebook).

    python -m scripts.fetch_data          (run from backend/)

Steps: get the NIFTY 100 constituent list from NSE (falls back to a local copy if
NSE blocks the request) -> download adjusted close + volume for 2022-2025 ->
compute the 7 features on the clustering period -> write data/raw/*.csv.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

from app import config
from app.data.features import compute_features, slice_period

log = logging.getLogger(__name__)


def get_nifty100_list(fallback_dir: Path = config.RAW_DATA_DIR) -> pd.DataFrame:
    """Download the official constituent list; fall back to data/raw/ind_nifty100list.csv."""
    import requests

    try:
        resp = requests.get(config.NIFTY100_LIST_URL,
                            headers={"User-Agent": config.BROWSER_USER_AGENT}, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        log.info("Downloaded NIFTY 100 list from NSE (%d rows)", len(df))
    except Exception as exc:  # NSE often blocks scripted requests
        local = fallback_dir / config.NIFTY_LIST_FILE
        if not local.exists():
            raise RuntimeError(f"NSE request failed ({exc}) and no local {local} found") from exc
        log.warning("NSE request failed (%s); using local %s", exc, local)
        df = pd.read_csv(local)
    df.columns = [c.strip() for c in df.columns]
    df["Ticker"] = df["Symbol"].astype(str).str.strip() + ".NS"
    return df[["Company Name", "Industry", "Symbol", "Ticker"]]


def fetch_all(out_dir: Path = config.RAW_DATA_DIR) -> dict:
    """Download everything and write the three CSVs. Returns a small summary dict."""
    import yfinance as yf

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = get_nifty100_list(out_dir)
    tickers = meta["Ticker"].tolist() + [config.INDEX_TICKER]

    # end is exclusive in yfinance, so add one day to include 31 Dec 2025.
    end = (pd.Timestamp(config.TEST_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=config.CLUSTER_START, end=end,
                      auto_adjust=True, progress=False, group_by="column")
    prices = raw["Close"].copy()
    volumes = raw["Volume"].copy()
    prices.index.name = volumes.index.name = "Date"

    cl_p = slice_period(prices, config.CLUSTER_START, config.CLUSTER_END)
    cl_v = slice_period(volumes, config.CLUSTER_START, config.CLUSTER_END)
    res = compute_features(cl_p, cl_v, meta["Ticker"].tolist())
    feats = meta.set_index("Ticker").join(res.features, how="inner").reset_index()
    feats = feats[["Company Name", "Industry", "Symbol", "Ticker"] + config.FEATURE_COLUMNS]

    feats.to_csv(out_dir / config.FEATURES_FILE, index=False)
    prices.to_csv(out_dir / config.PRICES_FILE)
    volumes.to_csv(out_dir / config.VOLUMES_FILE)
    return {"stocks": len(feats), "dropped": res.dropped, "out_dir": str(out_dir)}

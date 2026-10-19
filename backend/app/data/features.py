"""
The 7 behaviour features, computed from daily prices over the CLUSTERING period only.

Every formula here matches the project spec exactly (252 trading days/year, risk-free 6.5%).
The functions take an already-sliced price frame; `slice_period` is the single gate
that cuts the data to a date window, so leakage of 2025 data is impossible as long as
`compute_features` is only ever called with the clustering-period slice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app import config

log = logging.getLogger(__name__)


def slice_period(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Return rows whose Date index lies in [start, end] (inclusive)."""
    return frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns r_t = P_t / P_{t-1} - 1. Missing days stay NaN (no filling)."""
    return prices.pct_change(fill_method=None)


def annual_return(p: pd.Series) -> float:
    """(last / first) ^ (252 / trading days) - 1  -- compound annual growth rate."""
    p = p.dropna()
    trading_days = len(p)
    return float((p.iloc[-1] / p.iloc[0]) ** (config.TRADING_DAYS_PER_YEAR / trading_days) - 1)


def volatility(r: pd.Series) -> float:
    """Annualised volatility = std(daily returns) * sqrt(252)."""
    return float(r.dropna().std() * np.sqrt(config.TRADING_DAYS_PER_YEAR))


def beta_and_correlation(r_stock: pd.Series, r_index: pd.Series) -> tuple[float, float]:
    """
    Beta = cov(stock, index) / var(index); Correlation = corr(stock, index).
    Both are computed only on days where BOTH series have a return.
    """
    both = pd.concat([r_stock, r_index], axis=1).dropna()
    s, m = both.iloc[:, 0], both.iloc[:, 1]
    beta = float(s.cov(m) / m.var())
    corr = float(s.corr(m))
    return beta, corr


def max_drawdown(p: pd.Series) -> float:
    """Worst peak-to-trough fall: min(price / running max - 1). Always <= 0."""
    p = p.dropna()
    return float((p / p.cummax() - 1).min())


def sharpe_ratio(ann_return: float, vol: float) -> float:
    """(annual return - risk-free) / volatility."""
    return float((ann_return - config.RISK_FREE_RATE) / vol)


def log_traded_value(p: pd.Series, v: pd.Series) -> float:
    """log(mean of daily price x volume) - a size/liquidity proxy (rupees, log scale)."""
    traded = (p * v).dropna()
    traded = traded[traded > 0]
    return float(np.log(traded.mean()))


@dataclass
class FeatureResult:
    features: pd.DataFrame                      # index = ticker, columns = FEATURE_COLUMNS
    dropped: list[dict] = field(default_factory=list)   # [{"ticker":..., "reason":...}]


def compute_features(prices: pd.DataFrame, volumes: pd.DataFrame | None,
                     tickers: list[str]) -> FeatureResult:
    """
    Compute the 7 features for each ticker.

    `prices` / `volumes` MUST already be sliced to the clustering period by the caller.
    A stock missing more than MAX_MISSING_FRACTION of the index's trading days is dropped.
    """
    if config.INDEX_TICKER not in prices.columns:
        raise ValueError(f"Price file must contain the index column {config.INDEX_TICKER}")

    # "Trading days" = days the NIFTY 50 traded in this window.
    index_px = prices[config.INDEX_TICKER].dropna()
    trading_days = index_px.index
    rets = daily_returns(prices.loc[trading_days])
    r_index = rets[config.INDEX_TICKER]

    rows, dropped = {}, []
    for t in tickers:
        if t not in prices.columns:
            dropped.append({"ticker": t, "reason": "no price column"})
            continue
        p = prices.loc[trading_days, t]
        missing = float(p.isna().mean())
        if missing > config.MAX_MISSING_FRACTION:
            dropped.append({"ticker": t,
                            "reason": f"missing {missing:.0%} of trading days (> {config.MAX_MISSING_FRACTION:.0%})"})
            continue
        r = rets[t]
        ar = annual_return(p)
        vol = volatility(r)
        beta, corr = beta_and_correlation(r, r_index)
        if volumes is not None and t in volumes.columns:
            ltv = log_traded_value(p, volumes.reindex(trading_days)[t])
        else:
            ltv = np.nan
        rows[t] = {
            "Annual_Return": ar,
            "Volatility": vol,
            "Beta": beta,
            "Correlation": corr,
            "Max_Drawdown": max_drawdown(p),
            "Sharpe_Ratio": sharpe_ratio(ar, vol),
            "Log_Traded_Value": ltv,
        }

    for d in dropped:
        log.info("Dropped %s: %s", d["ticker"], d["reason"])
    feats = pd.DataFrame.from_dict(rows, orient="index")[config.FEATURE_COLUMNS] if rows \
        else pd.DataFrame(columns=config.FEATURE_COLUMNS)
    return FeatureResult(features=feats, dropped=dropped)

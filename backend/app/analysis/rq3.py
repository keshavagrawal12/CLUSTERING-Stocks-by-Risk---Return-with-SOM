"""
RQ3 - Does diversifying across SOM clusters lower risk more than diversifying across sectors?

Out-of-sample test: the clusters were learned on 2022-2024; here we only look at 2025.

Monte Carlo (seeded), RQ3_N_PORTFOLIOS portfolios per rule, each with k stocks
(k = number of SOM clusters), equal-weighted:
  * Cluster-diversified: exactly one random stock from each SOM cluster.
  * Sector-diversified : k different industries chosen at random, one random stock from each.

Portfolio = buy-and-hold with equal starting weights: value_t = mean_i( P_i,t / P_i,first ).
Metrics on 2025 daily returns: annualised volatility, annual return, Sharpe, max drawdown.
The two volatility distributions are compared with a Mann-Whitney U test (no normality
assumption - portfolio volatilities are skewed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from app import config
from app.data.features import slice_period

METRICS = ["volatility", "annual_return", "sharpe", "max_drawdown"]


def test_period_prices(prices: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """
    2025 prices for the eligible stocks. Stocks missing > RQ3_MAX_MISSING_FRACTION of 2025
    trading days are excluded; small gaps are forward-filled (a missing quote = price unchanged).
    """
    test = slice_period(prices, config.TEST_START, config.TEST_END)
    days = test[config.INDEX_TICKER].dropna().index
    test = test.loc[days]
    keep, excluded = [], []
    for t in tickers:
        if t not in test.columns:
            excluded.append({"ticker": t, "reason": "no 2025 prices"})
            continue
        miss = float(test[t].isna().mean())
        if miss > config.RQ3_MAX_MISSING_FRACTION:
            excluded.append({"ticker": t, "reason": f"missing {miss:.0%} of 2025 days"})
        else:
            keep.append(t)
    px = test[keep].ffill().bfill()
    return px, excluded


def _metrics_from_values(values: np.ndarray) -> dict[str, np.ndarray]:
    """values: (P, T) portfolio value paths starting at 1. Returns per-portfolio metrics."""
    rets = values[:, 1:] / values[:, :-1] - 1
    days = values.shape[1]
    vol = rets.std(axis=1, ddof=1) * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    ann = (values[:, -1] / values[:, 0]) ** (config.TRADING_DAYS_PER_YEAR / days) - 1
    sharpe = (ann - config.RISK_FREE_RATE) / vol
    mdd = (values / np.maximum.accumulate(values, axis=1) - 1).min(axis=1)
    return {"volatility": vol, "annual_return": ann, "sharpe": sharpe, "max_drawdown": mdd}


def _summary(x: np.ndarray, rng: np.random.Generator) -> dict:
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    lo = float(x[x >= q1 - 1.5 * iqr].min())          # Tukey whiskers for the box plot
    hi = float(x[x <= q3 + 1.5 * iqr].max())
    sample = rng.choice(x, size=min(config.RQ3_BOX_SAMPLE, len(x)), replace=False)
    return {"median": float(med), "q1": float(q1), "q3": float(q3), "mean": float(x.mean()),
            "min": float(x.min()), "max": float(x.max()), "whisker_low": lo, "whisker_high": hi,
            "sample": [float(v) for v in sample]}


def run_rq3(prices: pd.DataFrame, tickers: list[str], clusters: np.ndarray, sectors: np.ndarray,
            n_portfolios: int = config.RQ3_N_PORTFOLIOS, seed: int = config.RQ3_SEED) -> dict:
    px, excluded = test_period_prices(prices, tickers)
    pos = {t: i for i, t in enumerate(tickers)}
    eligible = list(px.columns)
    cl = np.array([clusters[pos[t]] for t in eligible])
    sec = np.array([sectors[pos[t]] for t in eligible])

    # Each stock's growth path in 2025, starting at 1.0.
    growth = (px / px.iloc[0]).to_numpy().T                  # (stocks, days)

    cluster_ids = np.unique(cl)
    industries = np.unique(sec)
    k = len(cluster_ids)
    if len(industries) < k:
        raise ValueError(f"Only {len(industries)} industries but k = {k} clusters: "
                         "cannot build sector-diversified portfolios of the same size.")
    by_cluster = {c: np.where(cl == c)[0] for c in cluster_ids}
    by_sector = {s: np.where(sec == s)[0] for s in industries}

    rng = np.random.default_rng(seed)
    cluster_picks = np.array([[rng.choice(by_cluster[c]) for c in cluster_ids]
                              for _ in range(n_portfolios)])
    sector_picks = np.array([[rng.choice(by_sector[s]) for s in rng.choice(industries, k, replace=False)]
                             for _ in range(n_portfolios)])

    # Equal-weight buy-and-hold value path = average of the constituents' growth paths.
    v_cluster = growth[cluster_picks].mean(axis=1)            # (P, days)
    v_sector = growth[sector_picks].mean(axis=1)
    m_cluster = _metrics_from_values(v_cluster)
    m_sector = _metrics_from_values(v_sector)

    summ_rng = np.random.default_rng(seed + 1)
    dists = {m: {"cluster": _summary(m_cluster[m], summ_rng), "sector": _summary(m_sector[m], summ_rng)}
             for m in METRICS}

    # Two-sided test: are the volatility distributions different? (also report one-sided)
    u_two = mannwhitneyu(m_cluster["volatility"], m_sector["volatility"], alternative="two-sided")
    u_less = mannwhitneyu(m_cluster["volatility"], m_sector["volatility"], alternative="less")
    u_sharpe = mannwhitneyu(m_cluster["sharpe"], m_sector["sharpe"], alternative="two-sided")

    # Median portfolio of each rule (by volatility) for the cumulative-return chart.
    def median_idx(v: np.ndarray) -> int:
        return int(np.argsort(v)[len(v) // 2])

    ic, is_ = median_idx(m_cluster["volatility"]), median_idx(m_sector["volatility"])
    nifty = slice_period(prices[[config.INDEX_TICKER]], config.TEST_START, config.TEST_END)
    nifty = nifty.loc[px.index, config.INDEX_TICKER].ffill()
    curves = {
        "dates": [d.strftime("%Y-%m-%d") for d in px.index],
        "cluster": (v_cluster[ic] - 1).tolist(),
        "sector": (v_sector[is_] - 1).tolist(),
        "nifty": (nifty / nifty.iloc[0] - 1).tolist(),
        "cluster_members": [eligible[j] for j in cluster_picks[ic]],
        "sector_members": [eligible[j] for j in sector_picks[is_]],
    }

    med_c = dists["volatility"]["cluster"]["median"]
    med_s = dists["volatility"]["sector"]["median"]
    p = float(u_two.pvalue)
    return {
        "k": int(k),
        "n_portfolios": int(n_portfolios),
        "n_eligible": len(eligible),
        "excluded": excluded,
        "period": [config.TEST_START, config.TEST_END],
        "trading_days": int(len(px)),
        "distributions": dists,
        "median_vol_diff": float(med_c - med_s),
        "mann_whitney": {"u": float(u_two.statistic), "p_two_sided": p,
                         "p_cluster_lower": float(u_less.pvalue),
                         "sharpe_p_two_sided": float(u_sharpe.pvalue), "alpha": config.RQ3_ALPHA},
        "curves": curves,
        "verdict": verdict(med_c, med_s, p,
                           dists["sharpe"]["cluster"]["median"], dists["sharpe"]["sector"]["median"]),
    }


def verdict(med_c: float, med_s: float, p: float, sh_c: float, sh_s: float) -> str:
    """One plain-English sentence built only from the computed numbers."""
    diff_pp = (med_c - med_s) * 100
    sig = p < config.RQ3_ALPHA
    if sig and diff_pp < 0:
        head = (f"Cluster-diversified portfolios had lower 2025 volatility than sector-diversified ones "
                f"(median {med_c:.1%} vs {med_s:.1%}, {abs(diff_pp):.1f} pp lower, p = {p:.3g})")
    elif sig and diff_pp > 0:
        head = (f"Cluster-diversified portfolios were more volatile in 2025 than sector-diversified ones "
                f"(median {med_c:.1%} vs {med_s:.1%}, p = {p:.3g}), so clustering did not help reduce risk")
    else:
        head = (f"No significant difference in 2025 volatility between cluster- and sector-diversified "
                f"portfolios (median {med_c:.1%} vs {med_s:.1%}, p = {p:.3g})")
    return f"{head}; median Sharpe was {sh_c:.2f} vs {sh_s:.2f}."

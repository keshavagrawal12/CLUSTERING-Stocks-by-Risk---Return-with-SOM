import numpy as np
import pandas as pd
import pytest

from app import config
from app.data.features import compute_features, slice_period
from app.data.loader import build_dataset
from tests.conftest import INDUSTRIES, make_frames


def test_beta_and_correlation_recovered():
    """Prices built with r = beta*r_m + noise must give back beta (and the theoretical corr)."""
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2022-01-03", "2024-12-31")
    r_m = rng.normal(0, 0.01, len(dates))
    beta, noise = 1.4, 0.008
    r_s = beta * r_m + rng.normal(0, noise, len(dates))
    prices = pd.DataFrame({config.INDEX_TICKER: 100 * np.cumprod(1 + r_m),
                           "X.NS": 100 * np.cumprod(1 + r_s)}, index=dates)
    feats = compute_features(prices, None, ["X.NS"]).features.loc["X.NS"]
    expected_corr = beta * 0.01 / np.sqrt(beta ** 2 * 0.01 ** 2 + noise ** 2)
    assert feats["Beta"] == pytest.approx(beta, abs=0.05)
    assert feats["Correlation"] == pytest.approx(expected_corr, abs=0.03)


def test_feature_formulas_on_tiny_series():
    dates = pd.bdate_range("2022-01-03", periods=5)
    p = pd.DataFrame({config.INDEX_TICKER: [100, 101, 102, 101, 103],
                      "A.NS": [10, 12, 9, 11, 12]}, index=dates, dtype=float)
    v = pd.DataFrame({"A.NS": [1, 2, 3, 4, 5]}, index=dates, dtype=float)
    f = compute_features(p, v, ["A.NS"]).features.loc["A.NS"]
    assert f["Annual_Return"] == pytest.approx((12 / 10) ** (252 / 5) - 1)
    r = p["A.NS"].pct_change().dropna()
    assert f["Volatility"] == pytest.approx(r.std() * np.sqrt(252))
    assert f["Max_Drawdown"] == pytest.approx(9 / 12 - 1)
    assert f["Sharpe_Ratio"] == pytest.approx((f["Annual_Return"] - 0.065) / f["Volatility"])
    assert f["Log_Traded_Value"] == pytest.approx(np.log(np.mean(np.array([10, 24, 27, 44, 60.0]))))


def test_short_history_stocks_are_dropped():
    feats, pf, vf = make_frames(short_history=("BAN0.NS", "IT3.NS"))
    pf = slice_period(pf, config.CLUSTER_START, config.CLUSTER_END)
    res = compute_features(pf, vf, feats["Ticker"].tolist())
    dropped = {d["ticker"] for d in res.dropped}
    assert dropped == {"BAN0.NS", "IT3.NS"}
    assert "BAN0.NS" not in res.features.index
    assert len(res.features) == len(feats) - 2


def test_dataset_build_and_scaler(synthetic_csvs):
    ds = build_dataset(synthetic_csvs / config.FEATURES_FILE, synthetic_csvs / config.PRICES_FILE,
                       synthetic_csvs / config.VOLUMES_FILE)
    assert len(ds.tickers) == len(INDUSTRIES) * 8
    np.testing.assert_allclose(ds.X.mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(ds.X.std(axis=0), 1, atol=1e-9)
    np.testing.assert_allclose(ds.scaler.inverse_transform(ds.X), ds.features.to_numpy(), rtol=1e-9)

"""RQ3 must be out-of-sample: 2025 prices may never influence the features or the model."""

import numpy as np
import pandas as pd

from app import config
from app.analysis.rq3 import run_rq3
from app.data.loader import build_dataset
from tests.conftest import make_frames, to_bytes
import io


def _build(feats, pf, vf):
    return build_dataset(io.BytesIO(to_bytes(feats, False)), io.BytesIO(to_bytes(pf, True)),
                         io.BytesIO(to_bytes(vf, True)))


def test_no_2025_data_used_in_features():
    feats, pf, vf = make_frames()
    ds_a = _build(feats, pf, vf)

    # Wreck every 2025 price and volume. If ANY 2025 row leaked into the features, they'd change.
    pf2, vf2 = pf.copy(), vf.copy()
    test_rows = pf2.index >= pd.Timestamp(config.TEST_START)
    assert test_rows.any()
    pf2.loc[test_rows] = pf2.loc[test_rows] * np.random.default_rng(0).uniform(0.1, 10, pf2.loc[test_rows].shape)
    vf2.loc[test_rows] = 1.0
    ds_b = _build(feats, pf2, vf2)

    pd.testing.assert_frame_equal(ds_a.features, ds_b.features)
    np.testing.assert_array_equal(ds_a.X, ds_b.X)


def test_rq3_uses_only_2025():
    feats, pf, vf = make_frames()
    ds = _build(feats, pf, vf)
    rng = np.random.default_rng(0)
    clusters = rng.integers(0, 3, len(ds.tickers))
    out = run_rq3(ds.prices, ds.tickers, clusters, ds.industries, n_portfolios=200)
    assert out["curves"]["dates"][0] >= config.TEST_START
    assert out["curves"]["dates"][-1] <= config.TEST_END
    assert out["k"] == 3
    for m in ("volatility", "sharpe", "annual_return", "max_drawdown"):
        d = out["distributions"][m]
        assert d["cluster"]["q1"] <= d["cluster"]["median"] <= d["cluster"]["q3"]
    assert 0 <= out["mann_whitney"]["p_two_sided"] <= 1
    assert isinstance(out["verdict"], str) and len(out["verdict"]) > 20

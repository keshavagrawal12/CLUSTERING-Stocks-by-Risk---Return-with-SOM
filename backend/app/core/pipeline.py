"""
The end-to-end pipeline for one run, plus the functions that turn a finished run into
the JSON payloads the frontend needs (map, clusters, RQ1, RQ2, RQ3).

Every number the UI shows is produced here from the loaded data - nothing is invented.
"""

from __future__ import annotations

import json
import logging
import math
import time

import numpy as np
import pandas as pd

from app import config
from app.analysis import rq1 as rq1_mod
from app.analysis.rq2 import find_misfits
from app.analysis.rq3 import run_rq3
from app.core.state import AppState, Run
from app.models import autoencoder as ae_mod
from app.models.clustering import cluster_neurons, kmeans_baseline
from app.models.lvq import evaluate_lvq
from app.models.som import SelfOrganizingMap

log = logging.getLogger(__name__)

SOM_PARAM_KEYS = ["rows", "cols", "sigma0", "eta0", "ordering_iters", "convergence_iters", "init", "seed"]


def clean(obj):
    """Make a payload JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _r(a: np.ndarray, nd: int = 4) -> list:
    return np.round(np.asarray(a, dtype=float), nd).tolist()


# ---------------------------------------------------------------------------- autoencoder
def get_autoencoder(state: AppState, ae_params: dict | None = None):
    """Train (or reuse) the autoencoder on the standardised features."""
    if not ae_mod.AVAILABLE:
        raise RuntimeError("Autoencoder disabled: TensorFlow is not installed "
                           "(needs Python 3.10-3.12). Raw-feature mode still works.")
    ae_params = ae_params or {}
    key = tuple(sorted(ae_params.items()))
    with state.lock:
        cached = state.ae_cache.get(key)
    if cached is not None:
        return cached
    res = ae_mod.train_autoencoder(state.dataset.X, **ae_params)
    with state.lock:
        state.ae_cache[key] = res
    return res


# ---------------------------------------------------------------------------- training
def train_run(state: AppState, run: Run) -> None:
    """Runs in a background thread. Appends progress dicts to run.progress as it goes."""
    ds = state.dataset
    p = run.params
    try:
        X = ds.X
        if p["input_mode"] == "autoencoder":
            run.status = "autoencoder"
            run.progress.append({"type": "status", "status": "autoencoder",
                                 "message": "Training autoencoder (7 -> 3)..."})
            run.autoencoder = get_autoencoder(state, p.get("autoencoder") or {})
            X = run.autoencoder.latent
            run.progress.append({"type": "autoencoder", "epochs_run": run.autoencoder.epochs_run,
                                 "best_epoch": run.autoencoder.best_epoch,
                                 "train_loss": _r(run.autoencoder.train_loss, 5),
                                 "val_loss": _r(run.autoencoder.val_loss, 5)})
        run.X_used = X
        som = SelfOrganizingMap(rows=p["rows"], cols=p["cols"], dim=X.shape[1], sigma0=p["sigma0"],
                                eta0=p["eta0"], ordering_iters=p["ordering_iters"],
                                convergence_iters=p["convergence_iters"], init=p["init"], seed=p["seed"])
        run.som = som
        run.status = "training"
        total = p["ordering_iters"] + p["convergence_iters"]
        for cp in som.train_iter(X):
            # At a checkpoint som.weights == cp.weights, so we can derive the views directly.
            run.progress.append({
                "type": "checkpoint", "iteration": cp.iteration, "total": total, "phase": cp.phase,
                "eta": cp.eta, "sigma": cp.sigma, "qe": cp.qe, "te": cp.te,
                "umatrix": _r(som.u_matrix()), "hits": som.hit_map(X).tolist(),
                "weights": _r(cp.weights, 3),
            })
        run.status = "clustering"
        run.bmus = som.bmus(X)
        run.clustering = cluster_neurons(som.weights, run.bmus, seed=p["seed"])
        run.status = "done"
        run.finished_at = time.time()
        run.progress.append({"type": "done", "run_id": run.id, "k": run.clustering.k,
                             "qe": som.quantization_error(X), "te": som.topographic_error(X)})
        _save_to_disk(state, run)
    except Exception as exc:  # report to the UI instead of dying silently in the thread
        log.exception("Run %s failed", run.id)
        run.status = "error"
        run.error = str(exc)
        run.progress.append({"type": "error", "message": str(exc)})


def train_sync(state: AppState, params: dict, hidden: bool = False) -> Run:
    run = state.new_run(params, hidden=hidden)
    train_run(state, run)
    return run


def _save_to_disk(state: AppState, run: Run) -> None:
    """Optional JSON copy of a finished run (map + clusters + learning curve) in data/cache/."""
    try:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"run_id": run.id, "params": run.params,
                   "history": [{k: v for k, v in c.items() if k in ("iteration", "phase", "eta", "sigma", "qe", "te")}
                               for c in run.progress if c.get("type") == "checkpoint"],
                   "map": map_payload(state, run), "clusters": clusters_payload(state, run)}
        (config.CACHE_DIR / f"run_{run.id}.json").write_text(json.dumps(clean(payload)))
    except Exception:  # disk cache is a convenience; never fail a run because of it
        log.warning("Could not write run cache", exc_info=True)


# ---------------------------------------------------------------------------- payloads
def _real_weights(state: AppState, run: Run) -> np.ndarray:
    """Neuron weights in real units (rows, cols, 7). Latent-space maps are decoded first."""
    W = run.som.weights.reshape(-1, run.som.dim)
    if run.params["input_mode"] == "autoencoder":
        W = run.autoencoder.decode(W)               # latent -> standardised 7 features
    return state.dataset.scaler.inverse_transform(W).reshape(run.som.rows, run.som.cols, -1)


def map_payload(state: AppState, run: Run) -> dict:
    if "map" in run.cache:
        return run.cache["map"]
    ds, som = state.dataset, run.som
    real_w = _real_weights(state, run)
    stocks = [{"ticker": t, "symbol": str(ds.meta.loc[t, "Symbol"]), "company": str(ds.meta.loc[t, "Company Name"]),
               "industry": str(ds.meta.loc[t, "Industry"]), "row": int(r), "col": int(c),
               "cluster": int(run.clustering.stock_labels[i])}
              for i, (t, (r, c)) in enumerate(zip(ds.tickers, run.bmus))]
    out = {
        "run_id": run.id, "rows": som.rows, "cols": som.cols, "input_mode": run.params["input_mode"],
        "hits": som.hit_map(run.X_used).tolist(),
        "umatrix": _r(som.u_matrix()),
        "component_planes": [{"feature": f, "unit": config.FEATURE_UNITS[f], "values": _r(real_w[:, :, i], 5)}
                             for i, f in enumerate(config.FEATURE_COLUMNS)],
        "planes_decoded": run.params["input_mode"] == "autoencoder",
        "neuron_clusters": run.clustering.neuron_labels.tolist(),
        "stocks": stocks,
        "qe": som.quantization_error(run.X_used), "te": som.topographic_error(run.X_used),
    }
    run.cache["map"] = clean(out)
    return run.cache["map"]


def clusters_payload(state: AppState, run: Run) -> dict:
    if "clusters" in run.cache:
        return run.cache["clusters"]
    ds, cl = state.dataset, run.clustering
    profiles = []
    for c in range(cl.k):
        idx = np.where(cl.stock_labels == c)[0]
        members = [ds.tickers[i] for i in idx]
        inds = pd.Series(ds.industries[idx]).value_counts()
        profiles.append({
            "cluster": c, "size": int(len(idx)), "members": members,
            "mean": {f: float(v) for f, v in ds.features.iloc[idx].mean().items()} if len(idx) else None,
            "median": {f: float(v) for f, v in ds.features.iloc[idx].median().items()} if len(idx) else None,
            "industries": [{"industry": str(k), "count": int(v)} for k, v in inds.items()],
            "neurons": int(np.sum(cl.neuron_labels == c)),
        })
    out = {"run_id": run.id, "k": cl.k, "silhouette": cl.silhouette,
           "stock_clusters": {t: int(cl.stock_labels[i]) for i, t in enumerate(ds.tickers)},
           "neuron_clusters": cl.neuron_labels.tolist(), "profiles": profiles}
    run.cache["clusters"] = clean(out)
    return run.cache["clusters"]


def companion(state: AppState, run: Run, mode: str) -> Run:
    """The same SOM hyperparameters trained on the other input (raw <-> autoencoder)."""
    if run.params["input_mode"] == mode:
        return run
    key = {k: run.params[k] for k in SOM_PARAM_KEYS}
    for r in state.runs.values():
        if r.done and r.params["input_mode"] == mode and {k: r.params[k] for k in SOM_PARAM_KEYS} == key:
            return r
    params = dict(run.params, input_mode=mode)
    r = train_sync(state, params, hidden=True)
    if not r.done:
        raise RuntimeError(r.error or "companion run failed")
    return r


def lvq_payload(state: AppState, params: dict | None = None) -> dict:
    params = params or {}
    key = tuple(sorted(params.items()))
    with state.lock:
        if key in state.lvq_cache:
            return state.lvq_cache[key]
    ev = evaluate_lvq(state.dataset.X, state.dataset.industries, **params)
    out = clean(ev.__dict__)
    with state.lock:
        state.lvq_cache[key] = out
    return out


def rq1_payload(state: AppState, run: Run) -> dict:
    if "rq1" in run.cache:
        return run.cache["rq1"]
    ds = state.dataset
    sectors = ds.industries
    variants = []

    raw = companion(state, run, "raw")
    variants.append({"name": "SOM (raw features)", "key": "som_raw", "k": raw.clustering.k,
                     **rq1_mod.agreement(raw.clustering.stock_labels, sectors),
                     "qe": raw.som.quantization_error(raw.X_used), "te": raw.som.topographic_error(raw.X_used),
                     "available": True})
    km = kmeans_baseline(ds.X, run.clustering.k, seed=run.params["seed"])
    variants.append({"name": "K-means baseline", "key": "kmeans", "k": run.clustering.k,
                     **rq1_mod.agreement(km, sectors), "qe": None, "te": None, "available": True})
    if ae_mod.AVAILABLE:
        try:
            aer = companion(state, run, "autoencoder")
            variants.append({"name": "Autoencoder + SOM", "key": "som_ae", "k": aer.clustering.k,
                             **rq1_mod.agreement(aer.clustering.stock_labels, sectors),
                             "qe": aer.som.quantization_error(aer.X_used),
                             "te": aer.som.topographic_error(aer.X_used), "available": True,
                             "ae_reconstruction_mse": aer.autoencoder.reconstruction_mse})
        except Exception as exc:
            variants.append({"name": "Autoencoder + SOM", "key": "som_ae", "available": False, "reason": str(exc)})
    else:
        variants.append({"name": "Autoencoder + SOM", "key": "som_ae", "available": False,
                         "reason": "TensorFlow not installed"})

    this = rq1_mod.agreement(run.clustering.stock_labels, sectors)
    out = {"run_id": run.id, "input_mode": run.params["input_mode"], "k": run.clustering.k,
           **this, "headline": rq1_mod.headline(this["ari"]),
           "variants": variants,
           "contingency": rq1_mod.contingency(run.clustering.stock_labels, sectors),
           "silhouette": run.clustering.silhouette,
           "lvq": lvq_payload(state)}
    run.cache["rq1"] = clean(out)
    return run.cache["rq1"]


def rq2_payload(state: AppState, run: Run) -> dict:
    if "rq2" in run.cache:
        return run.cache["rq2"]
    ds = state.dataset
    misfits = find_misfits(ds.tickers, ds.industries, run.bmus, run.clustering.stock_labels,
                           ds.features, ds.X, ds.meta["Company Name"].to_dict())
    n_mis = sum(m["cluster_mismatch"] for m in misfits)
    top = misfits[0] if misfits else None
    out = {"run_id": run.id, "misfits": misfits, "n_cluster_mismatch": n_mis, "n_ranked": len(misfits),
           "headline": (f"{n_mis} of {len(misfits)} stocks sit outside their sector's usual cluster; "
                        f"biggest outlier: {top['ticker'].replace('.NS', '')} ({top['industry']})") if top else
           "Not enough multi-stock sectors to rank misfits"}
    run.cache["rq2"] = clean(out)
    return run.cache["rq2"]


def rq3_payload(state: AppState, run: Run) -> dict:
    if "rq3" in run.cache:
        return run.cache["rq3"]
    ds = state.dataset
    out = run_rq3(ds.prices, ds.tickers, run.clustering.stock_labels, ds.industries)
    out["run_id"] = run.id
    run.cache["rq3"] = clean(out)
    return run.cache["rq3"]


def stock_detail(state: AppState, run: Run, ticker: str) -> dict:
    ds = state.dataset
    i = ds.tickers.index(ticker)
    industry = ds.industries[i]
    c = int(run.clustering.stock_labels[i])
    sec_idx = np.where(ds.industries == industry)[0]
    cl_idx = np.where(run.clustering.stock_labels == c)[0]
    return clean({
        "ticker": ticker, "company": str(ds.meta.loc[ticker, "Company Name"]), "industry": str(industry),
        "symbol": str(ds.meta.loc[ticker, "Symbol"]),
        "bmu": run.bmus[i].tolist(), "cluster": c,
        "features": ds.features.iloc[i].to_dict(),
        "z": dict(zip(config.FEATURE_COLUMNS, ds.X[i].tolist())),
        "sector_median": ds.features.iloc[sec_idx].median().to_dict(), "sector_size": int(len(sec_idx)),
        "cluster_mean": ds.features.iloc[cl_idx].mean().to_dict(), "cluster_size": int(len(cl_idx)),
    })

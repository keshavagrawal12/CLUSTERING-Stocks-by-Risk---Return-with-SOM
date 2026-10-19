"""
All HTTP endpoints. Each handler is thin: validate input, call the pipeline, return a
Pydantic model. The maths lives in app/models and app/analysis.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app import config
from app.api import schemas as S
from app.core import pipeline as P
from app.core.state import STATE, Run
from app.data.loader import DataError, load_from_bytes, load_from_dir
from app.models import autoencoder as ae_mod
from app.models.som import SelfOrganizingMap

router = APIRouter(prefix="/api")


# ------------------------------------------------------------------ helpers
def _need_data():
    if STATE.dataset is None:
        raise HTTPException(status_code=409, detail={
            "code": "no_data",
            "message": "No data loaded. Upload the three CSVs on the Data page, or place them in "
                       "backend/data/raw/ and reload, or run `python -m scripts.fetch_data`."})
    return STATE.dataset


def _get_run(run_id: str) -> Run:
    _need_data()
    run = STATE.runs.get(run_id)
    if run is None:
        raise HTTPException(404, detail={"code": "no_run", "message": f"Run {run_id} not found (train a map first)."})
    if run.status == "error":
        raise HTTPException(500, detail={"code": "run_failed", "message": run.error})
    if not run.done:
        raise HTTPException(409, detail={"code": "run_pending", "message": f"Run {run_id} is still {run.status}."})
    return run


def _status() -> dict:
    ds = STATE.dataset
    latest = STATE.latest_run()
    base = dict(
        loaded=ds is not None, load_error=STATE.load_error,
        autoencoder_available=ae_mod.AVAILABLE,
        autoencoder_reason=None if ae_mod.AVAILABLE else f"TensorFlow not available: {ae_mod.IMPORT_ERROR}",
        latest_run_id=latest.id if latest else None,
        defaults=S.TrainRequest().model_dump(),
    )
    if ds is not None:
        base.update(n_stocks=len(ds.tickers), n_industries=int(len(set(ds.industries))),
                    dropped=ds.dropped, source=ds.source, feature_check=P.clean(ds.feature_check),
                    suggested_neurons=int(round(5 * math.sqrt(len(ds.tickers)))))
    return base


# ------------------------------------------------------------------ data
@router.get("/status", response_model=S.StatusResponse)
def status():
    return _status()


@router.post("/data/upload", response_model=S.UploadResponse)
async def upload(features: UploadFile = File(...), prices: UploadFile = File(...), volumes: UploadFile = File(...)):
    try:
        ds = await asyncio.to_thread(load_from_bytes, await features.read(), await prices.read(), await volumes.read())
    except DataError as exc:
        raise HTTPException(400, detail={"code": "bad_data", "message": str(exc)})
    except Exception as exc:
        raise HTTPException(400, detail={"code": "bad_data", "message": f"Could not read files: {exc}"})
    STATE.set_dataset(ds)
    STATE.load_error = None
    return {**_status(), "message": f"Loaded {len(ds.tickers)} stocks"}


@router.post("/data/reload", response_model=S.StatusResponse)
def reload_from_folder():
    """Re-read backend/data/raw/ (e.g. after copying the CSVs in or running the fetch script)."""
    try:
        ds = load_from_dir()
        STATE.load_error = None if ds else "No CSVs found in backend/data/raw/"
    except Exception as exc:
        ds, STATE.load_error = None, str(exc)
    STATE.set_dataset(ds)
    return _status()


@router.get("/stocks", response_model=S.StocksResponse)
def stocks():
    ds = _need_data()
    rows = []
    for i, t in enumerate(ds.tickers):
        rows.append({"ticker": t, "symbol": str(ds.meta.loc[t, "Symbol"]),
                     "company": str(ds.meta.loc[t, "Company Name"]), "industry": str(ds.meta.loc[t, "Industry"]),
                     "features": ds.features.iloc[i].to_dict(),
                     "z": dict(zip(config.FEATURE_COLUMNS, ds.X[i].tolist()))})
    return P.clean({"stocks": rows, "industries": sorted(set(ds.industries)),
                    "features": config.FEATURE_COLUMNS, "feature_units": config.FEATURE_UNITS})


# ------------------------------------------------------------------ SOM training
@router.post("/som/train", response_model=S.TrainResponse)
def train(req: S.TrainRequest):
    _need_data()
    if req.input_mode == "autoencoder" and not ae_mod.AVAILABLE:
        raise HTTPException(400, detail={"code": "ae_unavailable",
                                         "message": "TensorFlow is not installed, so autoencoder mode is disabled."})
    params = req.model_dump()
    if params.get("autoencoder") is None:
        params["autoencoder"] = {}
    run = STATE.new_run(params)
    threading.Thread(target=P.train_run, args=(STATE, run), daemon=True).start()
    return {"run_id": run.id, "status": run.status}


@router.get("/som/train/{run_id}/stream")
async def stream(run_id: str):
    """
    Server-Sent Events: one `data: {...}` line per progress item. The frontend's
    EventSource receives checkpoints (iteration, eta, sigma, QE, TE, U-matrix, hits,
    weights) as soon as the training thread records them, then a final done/error event.
    """
    run = STATE.runs.get(run_id)
    if run is None:
        raise HTTPException(404, detail={"code": "no_run", "message": f"Run {run_id} not found"})

    async def events():
        i = 0
        while True:
            while i < len(run.progress):
                item = run.progress[i]
                i += 1
                yield f"event: {item['type']}\ndata: {json.dumps(P.clean(item))}\n\n"
                if item["type"] in ("done", "error"):
                    return
            await asyncio.sleep(0.05)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/runs", response_model=S.RunsResponse)
def runs():
    out = []
    for r in sorted(STATE.runs.values(), key=lambda r: r.created_at, reverse=True):
        if r.hidden:
            continue
        last = next((c for c in reversed(r.progress) if c.get("type") == "done"), {})
        out.append({"run_id": r.id, "status": r.status, "params": r.params, "created_at": r.created_at,
                    "finished_at": r.finished_at, "k": last.get("k"), "qe": last.get("qe"),
                    "te": last.get("te"), "error": r.error})
    latest = STATE.latest_run()
    return P.clean({"runs": out, "latest_run_id": latest.id if latest else None})


@router.get("/som/{run_id}/history")
def history(run_id: str):
    """All recorded checkpoints of a finished run (for replaying the training animation)."""
    run = _get_run(run_id)
    return P.clean({"run_id": run.id, "params": run.params, "progress": run.progress})


@router.get("/som/{run_id}/map", response_model=S.MapResponse)
def som_map(run_id: str):
    return P.map_payload(STATE, _get_run(run_id))


@router.get("/som/{run_id}/clusters", response_model=S.ClustersResponse)
def som_clusters(run_id: str):
    return P.clusters_payload(STATE, _get_run(run_id))


@router.get("/som/{run_id}/stock/{ticker}", response_model=S.StockDetail)
def som_stock(run_id: str, ticker: str):
    run = _get_run(run_id)
    if ticker not in STATE.dataset.tickers:
        raise HTTPException(404, detail={"code": "no_stock", "message": f"{ticker} not in dataset"})
    return P.stock_detail(STATE, run, ticker)


@router.get("/som/{run_id}/rq1", response_model=S.RQ1Response)
async def som_rq1(run_id: str):
    run = _get_run(run_id)
    return await asyncio.to_thread(P.rq1_payload, STATE, run)


@router.get("/som/{run_id}/rq2", response_model=S.RQ2Response)
def som_rq2(run_id: str):
    return P.rq2_payload(STATE, _get_run(run_id))


@router.get("/som/{run_id}/rq3", response_model=S.RQ3Response)
async def som_rq3(run_id: str):
    run = _get_run(run_id)
    try:
        return await asyncio.to_thread(P.rq3_payload, STATE, run)
    except ValueError as exc:
        raise HTTPException(400, detail={"code": "rq3_failed", "message": str(exc)})


# ------------------------------------------------------------------ teaching step
@router.post("/som/step", response_model=S.StepResponse)
def som_step(req: S.StepRequest):
    """
    ONE iteration of Kohonen learning on ONE stock, for the 'How it works' demo.
    Uses exactly the same SelfOrganizingMap methods as real training.
    """
    ds = _need_data()
    if req.ticker not in ds.tickers:
        raise HTTPException(404, detail={"code": "no_stock", "message": f"{req.ticker} not in dataset"})
    som = SelfOrganizingMap(rows=req.rows, cols=req.cols, dim=ds.X.shape[1], init=req.init, seed=req.seed)
    if req.weights is None:
        som.initialise(ds.X)
    else:
        w = np.asarray(req.weights, dtype=float)
        if w.shape != (req.rows, req.cols, ds.X.shape[1]):
            raise HTTPException(422, detail={"code": "bad_weights",
                                             "message": f"weights must have shape {(req.rows, req.cols, ds.X.shape[1])}"})
        som.weights = w
    x = ds.X[ds.tickers.index(req.ticker)]
    before = som.weights.copy()
    dist = som.distances(x)                      # every neuron's output
    bmu = som.bmu(x)                             # competition
    h = som.update(x, bmu, req.eta, req.sigma)   # cooperation + adaptation
    return P.clean({"ticker": req.ticker, "features": config.FEATURE_COLUMNS, "x": x, "bmu": list(bmu),
                    "distances": dist, "neighbourhood": h, "weights_before": before,
                    "weights_after": som.weights, "eta": req.eta, "sigma": req.sigma})


# ------------------------------------------------------------------ LVQ / autoencoder
@router.post("/lvq/evaluate", response_model=S.LVQResponse)
async def lvq(req: S.LVQRequest):
    _need_data()
    return await asyncio.to_thread(P.lvq_payload, STATE, req.model_dump())


@router.post("/autoencoder/train", response_model=S.AutoencoderResponse)
async def autoencoder(req: S.AutoencoderParams):
    ds = _need_data()
    if not ae_mod.AVAILABLE:
        raise HTTPException(400, detail={"code": "ae_unavailable",
                                         "message": "TensorFlow is not installed, so the autoencoder is disabled."})
    res = await asyncio.to_thread(P.get_autoencoder, STATE, req.model_dump())
    return P.clean({"train_loss": res.train_loss, "val_loss": res.val_loss, "best_epoch": res.best_epoch,
                    "epochs_run": res.epochs_run, "reconstruction_mse": res.reconstruction_mse,
                    "latent_dim": res.latent.shape[1], "config": res.config,
                    "latent": {t: res.latent[i].tolist() for i, t in enumerate(ds.tickers)}})

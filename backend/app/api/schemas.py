"""
Pydantic models for every request and response.

Request models carry validation (e.g. grid size limits) so bad input gets a clear 422
instead of a crash deep inside NumPy. Response models document the JSON shape the
frontend relies on.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from app import config

Grid = list[list[float]]
IntGrid = list[list[int]]
Weights3D = list[list[list[float]]]
FeatureDict = dict[str, Optional[float]]


# ------------------------------------------------------------------ status / data
class DroppedStock(BaseModel):
    ticker: str
    reason: str


class StatusResponse(BaseModel):
    loaded: bool
    n_stocks: int = 0
    n_industries: int = 0
    dropped: list[DroppedStock] = []
    source: Optional[str] = None
    load_error: Optional[str] = None
    feature_check: dict[str, Optional[float]] = {}
    features: list[str] = config.FEATURE_COLUMNS
    feature_units: dict[str, str] = config.FEATURE_UNITS
    periods: dict[str, list[str]] = {
        "clustering": [config.CLUSTER_START, config.CLUSTER_END],
        "test": [config.TEST_START, config.TEST_END],
    }
    autoencoder_available: bool
    autoencoder_reason: Optional[str] = None
    latest_run_id: Optional[str] = None
    suggested_neurons: Optional[int] = None
    defaults: dict


class StockRow(BaseModel):
    ticker: str
    symbol: str
    company: str
    industry: str
    features: FeatureDict
    z: FeatureDict


class StocksResponse(BaseModel):
    stocks: list[StockRow]
    industries: list[str]
    features: list[str]
    feature_units: dict[str, str]


# ------------------------------------------------------------------ training
class AutoencoderParams(BaseModel):
    hidden: int = Field(config.AE_HIDDEN, ge=2, le=128)
    latent: int = Field(config.AE_LATENT, ge=2, le=6)
    dropout: float = Field(config.AE_DROPOUT, ge=0, le=0.8)
    l2: float = Field(config.AE_L2, ge=0, le=0.1)
    learning_rate: float = Field(config.AE_LEARNING_RATE, gt=0, le=0.1)
    epochs: int = Field(config.AE_EPOCHS, ge=10, le=5000)
    patience: int = Field(config.AE_PATIENCE, ge=1, le=500)
    seed: int = config.AE_SEED


class TrainRequest(BaseModel):
    rows: int = Field(config.SOM_ROWS, ge=2, le=20)
    cols: int = Field(config.SOM_COLS, ge=2, le=20)
    sigma0: float = Field(config.SOM_SIGMA0, gt=0, le=20)
    eta0: float = Field(config.SOM_ETA0, gt=0, le=1)
    ordering_iters: int = Field(config.SOM_ORDERING_ITERS, ge=10, le=100_000)
    convergence_iters: int = Field(config.SOM_CONVERGENCE_ITERS, ge=10, le=200_000)
    init: Literal["pca", "random"] = config.SOM_INIT
    input_mode: Literal["raw", "autoencoder"] = "raw"
    seed: int = config.SOM_SEED
    autoencoder: Optional[AutoencoderParams] = None


class TrainResponse(BaseModel):
    run_id: str
    status: str


class RunSummary(BaseModel):
    run_id: str
    status: str
    params: dict
    created_at: float
    finished_at: Optional[float] = None
    k: Optional[int] = None
    qe: Optional[float] = None
    te: Optional[float] = None
    error: Optional[str] = None


class RunsResponse(BaseModel):
    runs: list[RunSummary]
    latest_run_id: Optional[str] = None


# ------------------------------------------------------------------ map / clusters
class ComponentPlane(BaseModel):
    feature: str
    unit: str
    values: Grid


class MapStock(BaseModel):
    ticker: str
    symbol: str
    company: str
    industry: str
    row: int
    col: int
    cluster: int


class MapResponse(BaseModel):
    run_id: str
    rows: int
    cols: int
    input_mode: str
    hits: IntGrid
    umatrix: Grid
    component_planes: list[ComponentPlane]
    planes_decoded: bool
    neuron_clusters: IntGrid
    stocks: list[MapStock]
    qe: float
    te: float


class SilhouettePoint(BaseModel):
    k: int
    score: float


class ClusterProfile(BaseModel):
    cluster: int
    size: int
    members: list[str]
    mean: Optional[FeatureDict] = None
    median: Optional[FeatureDict] = None
    industries: list[dict]
    neurons: int


class ClustersResponse(BaseModel):
    run_id: str
    k: int
    silhouette: list[SilhouettePoint]
    stock_clusters: dict[str, int]
    neuron_clusters: IntGrid
    profiles: list[ClusterProfile]


class StockDetail(BaseModel):
    ticker: str
    company: str
    industry: str
    symbol: str
    bmu: list[int]
    cluster: int
    features: FeatureDict
    z: FeatureDict
    sector_median: FeatureDict
    sector_size: int
    cluster_mean: FeatureDict
    cluster_size: int


# ------------------------------------------------------------------ RQ1 / LVQ
class LVQRequest(BaseModel):
    eta0: float = Field(config.LVQ_ETA0, gt=0, le=1)
    epochs: int = Field(config.LVQ_EPOCHS, ge=1, le=1000)
    prototypes_per_class: int = Field(config.LVQ_PROTOTYPES_PER_CLASS, ge=1, le=5)
    folds: int = Field(config.LVQ_FOLDS, ge=2, le=10)
    seed: int = config.LVQ_SEED


class LVQResponse(BaseModel):
    accuracy: float
    accuracy_std: float
    fold_accuracies: list[float]
    baseline_accuracy: float
    baseline_class: str
    classes: list[str]
    confusion: list[list[int]]
    merged_into_other: list[str]
    n_samples: int
    folds: int


class Variant(BaseModel):
    name: str
    key: str
    available: bool
    k: Optional[int] = None
    ari: Optional[float] = None
    nmi: Optional[float] = None
    qe: Optional[float] = None
    te: Optional[float] = None
    reason: Optional[str] = None
    ae_reconstruction_mse: Optional[float] = None


class Contingency(BaseModel):
    clusters: list[int]
    industries: list[str]
    counts: list[list[int]]


class RQ1Response(BaseModel):
    run_id: str
    input_mode: str
    k: int
    ari: float
    nmi: float
    headline: str
    variants: list[Variant]
    contingency: Contingency
    silhouette: list[SilhouettePoint]
    lvq: LVQResponse


# ------------------------------------------------------------------ RQ2
class Misfit(BaseModel):
    rank: int
    ticker: str
    company: str
    industry: str
    sector_size: int
    bmu: list[int]
    sector_centre: list[float]
    grid_distance: float
    cluster: int
    sector_modal_cluster: int
    cluster_mismatch: bool
    features: FeatureDict
    sector_median: FeatureDict
    z_diff: FeatureDict
    explanation: str


class RQ2Response(BaseModel):
    run_id: str
    misfits: list[Misfit]
    n_cluster_mismatch: int
    n_ranked: int
    headline: str


# ------------------------------------------------------------------ RQ3
class DistSummary(BaseModel):
    median: float
    q1: float
    q3: float
    mean: float
    min: float
    max: float
    whisker_low: float
    whisker_high: float
    sample: list[float]


class RuleDists(BaseModel):
    cluster: DistSummary
    sector: DistSummary


class MannWhitney(BaseModel):
    u: float
    p_two_sided: float
    p_cluster_lower: float
    sharpe_p_two_sided: float
    alpha: float


class Curves(BaseModel):
    dates: list[str]
    cluster: list[float]
    sector: list[float]
    nifty: list[Optional[float]]
    cluster_members: list[str]
    sector_members: list[str]


class RQ3Response(BaseModel):
    run_id: str
    k: int
    n_portfolios: int
    n_eligible: int
    excluded: list[DroppedStock]
    period: list[str]
    trading_days: int
    distributions: dict[str, RuleDists]
    median_vol_diff: float
    mann_whitney: MannWhitney
    curves: Curves
    verdict: str


# ------------------------------------------------------------------ teaching step
class StepRequest(BaseModel):
    ticker: str
    weights: Optional[Weights3D] = Field(None, description="Current map weights (z-units). "
                                         "Omit to start from a freshly initialised map.")
    rows: int = Field(config.SOM_ROWS, ge=2, le=20)
    cols: int = Field(config.SOM_COLS, ge=2, le=20)
    eta: float = Field(config.SOM_ETA0, gt=0, le=1)
    sigma: float = Field(config.SOM_SIGMA0, gt=0, le=20)
    init: Literal["pca", "random"] = "random"
    seed: int = config.SOM_SEED


class StepResponse(BaseModel):
    ticker: str
    features: list[str]
    x: list[float]
    bmu: list[int]
    distances: Grid
    neighbourhood: Grid
    weights_before: Weights3D
    weights_after: Weights3D
    eta: float
    sigma: float


# ------------------------------------------------------------------ autoencoder
class AutoencoderResponse(BaseModel):
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int
    epochs_run: int
    reconstruction_mse: float
    latent_dim: int
    config: dict
    latent: dict[str, list[float]]


class UploadResponse(StatusResponse):
    message: str

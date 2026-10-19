"""
In-memory application state: the loaded dataset and every trained run.

A single AppState instance lives for the life of the server. Training happens in a
background thread and appends progress dicts to `Run.progress`; the SSE endpoint just
reads that list, so streaming needs no queues or extra infrastructure.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.data.loader import Dataset
from app.models.autoencoder import AutoencoderResult
from app.models.clustering import NeuronClustering
from app.models.som import SelfOrganizingMap


@dataclass
class Run:
    id: str
    params: dict                                   # SOM hyperparameters + input_mode
    status: str = "queued"                         # queued | autoencoder | training | clustering | done | error
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    progress: list[dict] = field(default_factory=list)      # streamed checkpoints
    som: SelfOrganizingMap | None = None
    X_used: np.ndarray | None = None               # the matrix the SOM was trained on
    bmus: np.ndarray | None = None                 # (N, 2)
    clustering: NeuronClustering | None = None
    autoencoder: AutoencoderResult | None = None
    hidden: bool = False                           # companion runs built for RQ1 comparisons
    cache: dict[str, Any] = field(default_factory=dict)     # memoised payloads (map, rq1, ...)

    @property
    def done(self) -> bool:
        return self.status == "done"


class AppState:
    def __init__(self) -> None:
        self.dataset: Dataset | None = None
        self.load_error: str | None = None
        self.runs: dict[str, Run] = {}
        self.ae_cache: dict[tuple, AutoencoderResult] = {}
        self.lvq_cache: dict[tuple, Any] = {}
        self.lock = threading.RLock()

    def set_dataset(self, ds: Dataset | None) -> None:
        """New data invalidates every run and cache (they were computed on the old data)."""
        with self.lock:
            self.dataset = ds
            self.runs.clear()
            self.ae_cache.clear()
            self.lvq_cache.clear()

    def new_run(self, params: dict, hidden: bool = False) -> Run:
        run = Run(id=uuid.uuid4().hex[:10], params=params, hidden=hidden)
        with self.lock:
            self.runs[run.id] = run
        return run

    def latest_run(self) -> Run | None:
        done = [r for r in self.runs.values() if r.done and not r.hidden]
        return max(done, key=lambda r: r.created_at) if done else None


STATE = AppState()

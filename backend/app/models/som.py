"""
Kohonen Self-Organizing Map, written from scratch in NumPy.
Syllabus: Unit 3 - "Neural Networks Based on Competition - Kohonen SOM" (Practical 8),
          Unit 1 - "Model of an artificial neuron".

THE NEURON (Unit 1 tie-in)
--------------------------
Each SOM neuron j is an artificial neuron with a weight vector w_j that has the SAME
length as the input (here 7 features). Unlike a perceptron it does not compute a
weighted sum followed by an activation; its "output" is the DISTANCE ||x - w_j||
between the input and its weights. So the weight vector acts as a PROTOTYPE: the
neuron fires most strongly (smallest distance) for inputs that look like its weights.

THE LEARNING LOOP (one iteration t)
-----------------------------------
  1. Competitive step   - pick a random stock x; find the Best Matching Unit (BMU),
                          the neuron whose weights are closest to x. Winner-takes-all.
  2. Cooperative step   - the BMU's grid neighbours are also allowed to learn, with a
                          strength given by a Gaussian of their GRID distance to the BMU:
                              h_j(t) = exp( -d_grid(j, bmu)^2 / (2 * sigma(t)^2) )
  3. Adaptive step      - Kohonen rule, moving each neuron towards x:
                              w_j <- w_j + eta(t) * h_j(t) * (x - w_j)
  eta(t) and sigma(t) shrink exponentially, so the map first organises globally and
  then fine-tunes locally (two phases: ORDERING then CONVERGENCE).

Unsupervised: the network never sees the sector labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from app import config


@dataclass
class SOMCheckpoint:
    """A snapshot recorded during training, used for learning curves and the animation."""
    iteration: int
    phase: str
    eta: float
    sigma: float
    qe: float
    te: float
    weights: np.ndarray | None = None    # (rows, cols, dim) copy; None if not stored


@dataclass
class SOMHistory:
    checkpoints: list[SOMCheckpoint] = field(default_factory=list)


def exp_decay(v0: float, t: int, n_iters: int, v_min: float) -> float:
    """
    Exponential decay from v0 to (roughly) v_min over n_iters iterations:
        v(t) = v0 * exp(-t / tau),  with tau = n_iters / ln(v0 / v_min)
    so that v(n_iters) == v_min exactly. Never goes below v_min.
    """
    if n_iters <= 0 or v0 <= v_min:
        return max(v0, v_min)
    tau = n_iters / np.log(v0 / v_min)
    return float(max(v0 * np.exp(-t / tau), v_min))


class SelfOrganizingMap:
    """
    A rectangular rows x cols Kohonen map.

    Grid size rule of thumb (Vesanto & Alhoniemi): total neurons ~ 5 * sqrt(N),
    where N = number of training samples. For ~100 stocks that is ~50 neurons, so
    the default 7 x 7 = 49 grid. Fewer neurons -> every neuron is a coarse cluster;
    many more -> most neurons are empty and the map just memorises stocks.
    """

    def __init__(self, rows: int = config.SOM_ROWS, cols: int = config.SOM_COLS,
                 dim: int = len(config.FEATURE_COLUMNS),
                 sigma0: float = config.SOM_SIGMA0, eta0: float = config.SOM_ETA0,
                 ordering_iters: int = config.SOM_ORDERING_ITERS,
                 convergence_iters: int = config.SOM_CONVERGENCE_ITERS,
                 conv_sigma0: float = config.SOM_CONV_SIGMA0,
                 conv_eta0: float = config.SOM_CONV_ETA0,
                 init: str = config.SOM_INIT, seed: int = config.SOM_SEED) -> None:
        self.rows, self.cols, self.dim = rows, cols, dim
        self.sigma0, self.eta0 = sigma0, eta0
        self.ordering_iters, self.convergence_iters = ordering_iters, convergence_iters
        # The fine-tuning phase must not start with a BIGGER radius/rate than phase 1 ended with.
        self.conv_sigma0 = min(conv_sigma0, sigma0)
        self.conv_eta0 = min(conv_eta0, eta0)
        self.init = init
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros((rows, cols, dim))
        # Grid coordinates of every neuron, shape (rows, cols, 2). Used for grid distances.
        rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        self.grid = np.stack([rr, cc], axis=-1).astype(float)

    # ------------------------------------------------------------------ initialisation
    def initialise(self, X: np.ndarray) -> None:
        """
        'random': each weight drawn uniformly inside the data's min/max range.
        'pca'   : spread the neurons on a flat sheet spanned by the first two principal
                  components. The map then starts already roughly ORDERED (neighbouring
                  neurons already hold similar prototypes), so the ordering phase only has
                  to bend the sheet, not untangle a random knot -> faster convergence and
                  the same result every run (it's deterministic).
        """
        if self.init == "random":
            lo, hi = X.min(axis=0), X.max(axis=0)
            self.weights = self.rng.uniform(lo, hi, size=(self.rows, self.cols, self.dim))
            return
        if self.init != "pca":
            raise ValueError("init must be 'random' or 'pca'")
        mean = X.mean(axis=0)
        # Principal components from the eigen-decomposition of the covariance matrix.
        cov = np.cov(X - mean, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)                 # ascending order
        order = np.argsort(eigvals)[::-1]
        pc1, pc2 = eigvecs[:, order[0]], eigvecs[:, order[1]]
        s1, s2 = np.sqrt(eigvals[order[0]]), np.sqrt(eigvals[order[1]])
        # Map grid rows to [-1, 1] along PC1 and columns along PC2, scaled by +-2 standard
        # deviations of each component so the sheet covers the bulk of the data.
        r = np.linspace(-1, 1, self.rows) if self.rows > 1 else np.zeros(1)
        c = np.linspace(-1, 1, self.cols) if self.cols > 1 else np.zeros(1)
        for i in range(self.rows):
            for j in range(self.cols):
                self.weights[i, j] = mean + 2 * s1 * r[i] * pc1 + 2 * s2 * c[j] * pc2

    # ------------------------------------------------------------------ building blocks
    def distances(self, x: np.ndarray) -> np.ndarray:
        """Output of every neuron for input x: Euclidean distance ||x - w_j||, shape (rows, cols)."""
        return np.linalg.norm(self.weights - x, axis=-1)

    def bmu(self, x: np.ndarray) -> tuple[int, int]:
        """
        COMPETITIVE STEP - winner-takes-all.
        The Best Matching Unit is the neuron with the smallest distance to x.
        """
        d = self.distances(x)
        idx = int(np.argmin(d))                      # flat index of the winner
        return divmod(idx, self.cols)                # -> (row, col)

    def neighbourhood(self, bmu: tuple[int, int], sigma: float) -> np.ndarray:
        """
        COOPERATIVE STEP - Gaussian neighbourhood on the GRID (not in feature space):
            h_j = exp( -d_grid(j, bmu)^2 / (2 sigma^2) )
        The BMU itself gets h = 1; neurons far away on the grid get h ~ 0.
        """
        d2 = np.sum((self.grid - np.array(bmu, dtype=float)) ** 2, axis=-1)
        return np.exp(-d2 / (2.0 * sigma ** 2))

    def update(self, x: np.ndarray, bmu: tuple[int, int], eta: float, sigma: float) -> np.ndarray:
        """
        ADAPTIVE STEP - the Kohonen update rule. Per neuron j it is literally:

            for each neuron j:
                w_j = w_j + eta * h_j * (x - w_j)

        Below it is vectorised over all neurons at once (same maths, much faster):
        h[..., None] broadcasts the (rows, cols) neighbourhood over the 7 features.
        Returns h so callers (the teaching step endpoint) can show it.
        """
        h = self.neighbourhood(bmu, sigma)
        self.weights += eta * h[..., None] * (x - self.weights)
        return h

    # ------------------------------------------------------------------ quality metrics
    def quantization_error(self, X: np.ndarray) -> float:
        """
        QE = average distance between each sample and its BMU's weight vector.
        "How well do the prototypes represent the data?"  Lower is better.
        """
        W = self.weights.reshape(-1, self.dim)
        d = np.linalg.norm(X[:, None, :] - W[None, :, :], axis=-1)   # (N, neurons)
        return float(d.min(axis=1).mean())

    def topographic_error(self, X: np.ndarray) -> float:
        """
        TE = fraction of samples whose best and second-best units are NOT grid neighbours
        (8-neighbourhood). "Is the map's topology preserved?"  Lower is better.
        """
        W = self.weights.reshape(-1, self.dim)
        d = np.linalg.norm(X[:, None, :] - W[None, :, :], axis=-1)
        best_two = np.argsort(d, axis=1)[:, :2]
        r1, c1 = np.divmod(best_two[:, 0], self.cols)
        r2, c2 = np.divmod(best_two[:, 1], self.cols)
        adjacent = (np.abs(r1 - r2) <= 1) & (np.abs(c1 - c2) <= 1)
        return float(1.0 - adjacent.mean())

    # ------------------------------------------------------------------ schedules
    def schedule(self, t: int) -> tuple[str, float, float]:
        """eta and sigma at global iteration t (two phases, each with its own exponential decay)."""
        if t < self.ordering_iters:
            # Ordering phase: decay from (sigma0, eta0) down to the convergence-phase start values.
            sigma = exp_decay(self.sigma0, t, self.ordering_iters, self.conv_sigma0)
            eta = exp_decay(self.eta0, t, self.ordering_iters, self.conv_eta0)
            return "ordering", eta, sigma
        tc = t - self.ordering_iters
        sigma = exp_decay(self.conv_sigma0, tc, self.convergence_iters, config.SOM_SIGMA_MIN)
        eta = exp_decay(self.conv_eta0, tc, self.convergence_iters, config.SOM_ETA_MIN)
        return "convergence", eta, sigma

    # ------------------------------------------------------------------ training
    def train_iter(self, X: np.ndarray, n_checkpoints: int = config.SOM_CHECKPOINTS,
                   store_weights: bool = True) -> Iterator[SOMCheckpoint]:
        """
        Train, yielding a checkpoint every few iterations (so the API can stream progress).
        The full training loop, step by step:
        """
        self.initialise(X)
        total = self.ordering_iters + self.convergence_iters
        every = max(1, total // max(1, n_checkpoints))

        def snap(t: int) -> SOMCheckpoint:
            phase, eta, sigma = self.schedule(min(t, total - 1))
            return SOMCheckpoint(t, phase, eta, sigma, self.quantization_error(X),
                                 self.topographic_error(X),
                                 self.weights.copy() if store_weights else None)

        yield snap(0)
        for t in range(total):
            phase, eta, sigma = self.schedule(t)          # current learning rate & radius
            x = X[self.rng.integers(len(X))]              # pick one random stock
            winner = self.bmu(x)                          # 1. competition
            self.update(x, winner, eta, sigma)            # 2.+3. cooperation + adaptation
            if (t + 1) % every == 0 or t + 1 == total:
                yield snap(t + 1)

    def fit(self, X: np.ndarray, **kw) -> SOMHistory:
        """Train to completion and return the recorded history."""
        hist = SOMHistory()
        for cp in self.train_iter(X, **kw):
            hist.checkpoints.append(cp)
        return hist

    # ------------------------------------------------------------------ outputs
    def bmus(self, X: np.ndarray) -> np.ndarray:
        """BMU (row, col) for every sample, shape (N, 2)."""
        return np.array([self.bmu(x) for x in X], dtype=int)

    def hit_map(self, X: np.ndarray) -> np.ndarray:
        """How many samples land on each neuron, shape (rows, cols)."""
        hits = np.zeros((self.rows, self.cols), dtype=int)
        for r, c in self.bmus(X):
            hits[r, c] += 1
        return hits

    def u_matrix(self) -> np.ndarray:
        """
        Unified distance matrix: for each neuron, the AVERAGE distance between its weight
        vector and those of its (up to 8) grid neighbours. High values = a "wall" between
        groups of neurons (a cluster boundary); low values = a valley inside a cluster.
        """
        U = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                dists = []
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.rows and 0 <= nj < self.cols:
                            dists.append(np.linalg.norm(self.weights[i, j] - self.weights[ni, nj]))
                U[i, j] = np.mean(dists) if dists else 0.0
        return U

    def component_planes(self) -> np.ndarray:
        """One (rows, cols) plane per input dimension: shape (dim, rows, cols)."""
        return np.transpose(self.weights, (2, 0, 1)).copy()

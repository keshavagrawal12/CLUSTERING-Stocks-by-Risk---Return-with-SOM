"""
Central configuration: every constant and default hyperparameter lives here.

Nothing numeric should be hard-coded anywhere else in the backend - if you see a
number in another module that isn't 0/1/2 or an array index, it should have come
from this file. That keeps the viva answer to "where does that number come from?"
always the same: config.py.
"""

from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BACKEND_DIR / "data" / "raw"          # put the three CSVs here
CACHE_DIR = BACKEND_DIR / "data" / "cache"            # trained runs saved as JSON

FEATURES_FILE = "nifty100_features.csv"
PRICES_FILE = "nifty100_prices.csv"
VOLUMES_FILE = "nifty100_volumes.csv"
NIFTY_LIST_FILE = "ind_nifty100list.csv"              # local fallback if NSE blocks us

NIFTY100_LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"
# NSE rejects requests that don't look like a browser, so we send a normal browser UA.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# --------------------------------------------------------------------------------------
# Periods - the clustering period builds the features; the test period is ONLY used
# for the out-of-sample portfolio test (RQ3). Never mix them.
# --------------------------------------------------------------------------------------
CLUSTER_START = "2022-01-01"
CLUSTER_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

INDEX_TICKER = "^NSEI"                                # NIFTY 50, the "market" for beta

# --------------------------------------------------------------------------------------
# Finance constants
# --------------------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252        # convention for annualising daily statistics
RISK_FREE_RATE = 0.065             # 6.5% p.a., roughly the Indian 10-year G-sec yield
MAX_MISSING_FRACTION = 0.10        # drop a stock missing >10% of trading days

# The 7 behaviour features, in the order the neural networks see them.
FEATURE_COLUMNS = [
    "Annual_Return",
    "Volatility",
    "Beta",
    "Correlation",
    "Max_Drawdown",
    "Sharpe_Ratio",
    "Log_Traded_Value",
]

# How each feature should be displayed in the UI (unit hint only, no maths).
FEATURE_UNITS = {
    "Annual_Return": "pct",
    "Volatility": "pct",
    "Beta": "x",
    "Correlation": "ratio",
    "Max_Drawdown": "pct",
    "Sharpe_Ratio": "ratio",
    "Log_Traded_Value": "log_inr",
}

# --------------------------------------------------------------------------------------
# Kohonen SOM defaults (Unit 3, Practical 8)
# --------------------------------------------------------------------------------------
# Grid size rule of thumb (Vesanto): total neurons ~ 5 * sqrt(N).
# With N ~ 100 stocks: 5 * sqrt(100) = 50 neurons  ->  7 x 7 = 49 is the closest square.
SOM_ROWS = 7
SOM_COLS = 7
SOM_SIGMA0 = 2.0                   # initial neighbourhood radius (in grid cells)
SOM_ETA0 = 0.5                     # initial learning rate
SOM_ORDERING_ITERS = 2_000         # phase 1: big sigma, high eta -> global ordering
SOM_CONVERGENCE_ITERS = 8_000      # phase 2: small sigma, low eta -> fine-tuning
# Convergence phase starts from these values (classic Kohonen advice: eta ~0.01-0.1,
# sigma shrinks to ~1 neuron so only the winner and its direct neighbours move).
SOM_CONV_ETA0 = 0.05
SOM_CONV_SIGMA0 = 1.0
SOM_SIGMA_MIN = 0.3                # sigma never decays below this (avoids divide-by-zero)
SOM_ETA_MIN = 0.001                # learning rate floor
SOM_INIT = "pca"                   # "pca" or "random"
SOM_SEED = 42
SOM_CHECKPOINTS = 40               # how many QE/TE + weight snapshots to record per run

# --------------------------------------------------------------------------------------
# Neuron -> cluster step (K-means over the trained SOM weight vectors)
# --------------------------------------------------------------------------------------
CLUSTER_K_MIN = 3
CLUSTER_K_MAX = 10
KMEANS_N_INIT = 20                 # K-means restarts (picks the best of 20 random starts)

# --------------------------------------------------------------------------------------
# LVQ1 defaults (Unit 3, Learning Vector Quantization)
# --------------------------------------------------------------------------------------
LVQ_ETA0 = 0.1
LVQ_EPOCHS = 60
LVQ_PROTOTYPES_PER_CLASS = 1
LVQ_INIT_NOISE = 0.05              # std of noise added to class means at init
LVQ_MIN_CLASS_SIZE = 3             # sectors with fewer stocks are merged into "Other"
LVQ_FOLDS = 5                      # stratified k-fold CV (small data -> 5 folds)
LVQ_SEED = 42

# --------------------------------------------------------------------------------------
# Autoencoder defaults (Units 4 & 5)
# --------------------------------------------------------------------------------------
AE_HIDDEN = 16
AE_LATENT = 3
AE_DROPOUT = 0.1
AE_L2 = 1e-4
AE_LEARNING_RATE = 1e-3
AE_EPOCHS = 400
AE_BATCH_SIZE = 16
AE_VAL_SPLIT = 0.2
AE_PATIENCE = 30                   # early stopping: stop after 30 epochs w/o val improvement
AE_SEED = 42

# --------------------------------------------------------------------------------------
# RQ2 misfits
# --------------------------------------------------------------------------------------
RQ2_MIN_SECTOR_SIZE = 2            # a sector needs >=2 stocks to define a "centre"

# --------------------------------------------------------------------------------------
# RQ3 Monte Carlo portfolio test
# --------------------------------------------------------------------------------------
RQ3_N_PORTFOLIOS = 2_000
RQ3_SEED = 42
RQ3_ALPHA = 0.05                   # significance level for the Mann-Whitney U test
RQ3_MAX_MISSING_FRACTION = 0.10    # a stock missing >10% of 2025 days is left out of RQ3
RQ3_BOX_SAMPLE = 400               # how many raw portfolio values to send for the violin/strip plot

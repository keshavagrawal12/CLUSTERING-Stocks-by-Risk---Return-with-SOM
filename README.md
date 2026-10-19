# Beyond Sector Labels

**Clustering NIFTY 100 stocks by risk and return with Self-Organizing Maps**
Course project · BDA404-5N Artificial Neural Networks and Deep Learning

Investors often diversify by picking stocks from different sectors, assuming that different sectors behave differently.
This project groups NIFTY 100 stocks by how they *actually* behave, using a **Kohonen Self-Organizing Map written from scratch in NumPy**.
It then asks three questions:

| | Research question | Where |
|---|---|---|
| **RQ1** | Do stocks grouped by risk/return behaviour match their official sectors? | *Clusters vs Sectors* page |
| **RQ2** | Which stocks behave differently from the rest of their sector? | *Misfits* page |
| **RQ3** | Does a portfolio diversified across SOM clusters have lower 2025 risk than one diversified across sectors? | *Portfolio test* page |

The app reports every result exactly as computed, including results that go against the hypothesis.

---

## Repository layout

```
backend/
  app/
    config.py            every constant & hyperparameter, commented
    data/                features.py (7 features) · loader.py · scaler.py · fetch.py (yfinance)
    models/              som.py (Kohonen SOM, NumPy) · lvq.py (LVQ1, NumPy)
                         autoencoder.py (Keras) · clustering.py (K-means over neurons, baseline)
    analysis/            rq1.py (ARI/NMI) · rq2.py (misfits) · rq3.py (2025 Monte Carlo)
    core/                state.py (in-memory runs) · pipeline.py (run → JSON payloads)
    api/                 routes.py (FastAPI) · schemas.py (Pydantic models)
    main.py
  scripts/fetch_data.py  rebuild the CSVs with yfinance
  tests/                 pytest suite (synthetic data lives ONLY here)
  data/raw/              put the three CSVs here
  requirements.txt
frontend/                React + TypeScript + Vite + Tailwind + D3 + Recharts
dev.sh / dev.ps1         start both servers with one command
```

---

## Setup

You need **Python 3.11 or 3.12** (TensorFlow does not support 3.13+ yet) and **Node.js 20+**.

### macOS

```bash
cd "ANN ESE PROJECT"
python3.11 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
cd frontend && npm install && cd ..
```

### Windows (PowerShell)

```powershell
cd "ANN ESE PROJECT"
py -3.11 -m venv backend\.venv
backend\.venv\Scripts\pip install -r backend\requirements.txt
cd frontend; npm install; cd ..
```

If TensorFlow fails to install, everything still works except the autoencoder. The app detects this and disables that option.

---

## Data

The app expects the three files produced by the Colab notebook:

| File | Contents |
|---|---|
| `nifty100_features.csv` | `Company Name, Industry, Symbol, Ticker` and the 7 features |
| `nifty100_prices.csv` | daily adjusted close, `Date` index, one column per ticker plus `^NSEI` |
| `nifty100_volumes.csv` | daily volumes, same shape |

There are three ways to load them:

1. **Copy them into `backend/data/raw/`.** They load automatically when the backend starts. If the backend is already running, use **Reload from folder** on the Data page.
2. **Upload them** on the **Data** page.
3. **Rebuild them from the internet:**
   ```bash
   cd backend
   .venv/bin/python -m scripts.fetch_data        # Windows: .venv\Scripts\python -m scripts.fetch_data
   ```
   This downloads the NIFTY 100 list from NSE with a browser User-Agent. If NSE blocks the request, it falls back to `backend/data/raw/ind_nifty100list.csv`. It then downloads 2022–2025 prices and volumes with yfinance.

**Leakage guard.** The backend does not trust the feature columns in the CSV. It **recomputes all 7 features from the price file, using only 2022-01-01 → 2024-12-31**, and shows how far its numbers are from the notebook's on the Data page. 2025 prices are used only by RQ3. A test changes every 2025 price and asserts that the features stay identical.

Stocks missing more than 10% of trading days in the clustering period are dropped, and the Data page lists them.
With no data loaded, every page shows an empty state; the app never makes up data.

---

## Running

**One command (macOS):**

```bash
./dev.sh
```

**One command (Windows):**

```powershell
powershell -ExecutionPolicy Bypass -File dev.ps1
```

Then open **http://localhost:5173**. The API runs on http://localhost:8000; interactive API docs are at http://localhost:8000/docs.

Or start the two servers separately:

```bash
cd backend && .venv/bin/python -m uvicorn app.main:app --port 8000
cd frontend && npm run dev
```

## Tests

```bash
cd backend
.venv/bin/python -m pytest -q
```

The suite has 26 tests. They cover:

- **SOM:**
  - BMU selection on a hand-made map
  - The Gaussian neighbourhood, and the Kohonen rule checked neuron by neuron
  - η and σ decay
  - QE falling on clustered data
  - Low TE on an ordered map, and a scrambled map scoring worse
  - Seed reproducibility
  - QE within ±50% of **MiniSom** on the same data
- **LVQ:**
  - Accuracy above 95% on separable data
  - Prototypes moving towards the input for the right class and away for the wrong class
  - Small classes merged into "Other"
- **Features:**
  - Beta and correlation recovered from synthetic prices with a known beta
  - Each formula checked on a tiny series by hand
  - Short-history stocks dropped
- **RQ3:**
  - No 2025 data reaches the features
  - RQ3 uses only 2025 dates
- **API:**
  - Every endpoint returns 200 on fixture data
  - Correct empty-state responses when nothing is loaded
  - Bad uploads rejected
  - The SSE stream, autoencoder mode, and a clean fallback when TensorFlow is missing

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/status` | Is data loaded, stock count, dropped stocks, feature sanity check, defaults |
| POST | `/api/data/upload` | Upload the three CSVs (multipart) |
| POST | `/api/data/reload` | Re-read `backend/data/raw/` |
| GET | `/api/stocks` | Stocks with raw (real-unit) and z-scored features |
| POST | `/api/som/train` | Start training in the background; returns `run_id` |
| GET | `/api/som/train/{run_id}/stream` | **Server-Sent Events**: iteration, η, σ, QE, TE, U-matrix, hit map, weights |
| GET | `/api/runs` | All runs (feeds the run picker) |
| GET | `/api/som/{run_id}/history` | Recorded checkpoints, for replaying the animation |
| GET | `/api/som/{run_id}/map` | Hit map, U-matrix, component planes (real units), BMUs, neuron clusters |
| GET | `/api/som/{run_id}/clusters` | Silhouette curve, chosen k, cluster per stock, cluster profiles |
| GET | `/api/som/{run_id}/stock/{ticker}` | One stock vs its sector median and cluster mean |
| GET | `/api/som/{run_id}/rq1` · `/rq2` · `/rq3` | Research-question results |
| POST | `/api/som/step` | Teaching mode: one Kohonen update for one stock |
| POST | `/api/lvq/evaluate` | LVQ1 cross-validation |
| POST | `/api/autoencoder/train` | Train the autoencoder and return loss curves and latent codes |

Runs are cached in memory. Each finished run is also written to `backend/data/cache/run_<id>.json`.

---

## Syllabus mapping

| Unit | Syllabus topic | Where it appears |
|---|---|---|
| Unit 1 | Model of artificial neuron, learning methods, taxonomy of NN architectures | SOM neuron as prototype; taxonomy panel (unsupervised / supervised / self-supervised) |
| Unit 3 | Neural networks based on competition, Kohonen SOM (Practical 8) | `models/som.py`: BMU, neighbourhood, Kohonen rule, decay, QE/TE, U-matrix |
| Unit 3 | Learning Vector Quantization | `models/lvq.py`: sector prediction from behaviour (RQ1) |
| Unit 4 | Regularization, dropout, initialization, gradient descent strategies | Autoencoder training (He init, dropout, L2, Adam, early stopping) |
| Unit 5 | Autoencoders (Practicals 13–14) | `models/autoencoder.py`: 3-D latent features fed to the SOM |

---

## How to explain this in a viva

### 1. What is a SOM neuron? (Unit 1)
Each of the 7×7 = 49 neurons stores a **weight vector with 7 numbers**, one per feature. A perceptron computes `f(Σwx + b)`. A SOM neuron instead outputs the **distance** `‖x − w‖` between the input and its weights. The weight vector is therefore a **prototype**: the typical stock behaviour that this neuron stands for.

### 2. Why 7×7?
The rule of thumb is about **5·√N neurons**. With N ≈ 100 stocks, 5·√100 = 50, so the closest square grid is 7×7 = 49. With fewer neurons, each one becomes a coarse cluster. With many more, most neurons stay empty.

### 3. Why standardise?
The SOM uses Euclidean distance. Without scaling, log traded value (~21) would swamp correlation (~0.4). After z-scoring, every feature has mean 0 and standard deviation 1, so each feature gets an equal vote. The scaler is fitted on 2022–24 only and kept, so the UI can convert prototypes back to %, × and ₹.

### 4. The training loop, line by line (`backend/app/models/som.py → train_iter`)

```python
self.initialise(X)                          # PCA (or random) start
for t in range(total):
    phase, eta, sigma = self.schedule(t)    # current learning rate & radius
    x = X[self.rng.integers(len(X))]        # pick ONE random stock
    winner = self.bmu(x)                    # 1. COMPETITION
    self.update(x, winner, eta, sigma)      # 2. COOPERATION + 3. ADAPTATION
```

- **`initialise`.** PCA initialisation lays the 49 neurons out as a flat sheet spanned by the first two principal components. The map then starts out already roughly ordered, so it converges faster and gives the same result every run. Random initialisation would first have to untangle itself.
- **`schedule(t)`.** Exponential decay `v(t) = v₀·exp(−t/τ)`, with τ chosen so the value reaches its end target exactly. It runs in two phases:
  - **Ordering** (2,000 iterations): σ goes from 2 to 1 and η from 0.5 to 0.05. The big radius lets the map organise globally.
  - **Convergence** (8,000 iterations): σ goes from 1 to 0.3 and η from 0.05 to 0.001. Only the winner and its direct neighbours move, which fine-tunes the prototypes.
- **`bmu(x)`.** Compute `‖x − w_j‖` for every neuron and take the argmin. This is **winner-takes-all competitive learning**.
- **`neighbourhood`.** `h_j = exp(−d_grid(j, bmu)² / 2σ²)`. The distance is measured **on the grid**, not in feature space. This is what makes neighbouring neurons hold similar prototypes, so the map preserves topology.
- **`update`.** The **Kohonen rule** `w_j ← w_j + η·h_j·(x − w_j)`. Every neuron moves a fraction η·h_j of the way towards the stock. The code is vectorised, but the per-neuron loop is written in the comment above it, and a test checks the two are identical.

You can show all of this live on the **How it works** page. Pick a stock and the BMU lights up, the Gaussian neighbourhood appears, and **Apply step** moves the weights with the real numbers substituted into the formula.

### 5. Quality metrics
- **QE (quantization error):** the average distance from each stock to its BMU's prototype. It measures how well the neurons represent the data.
- **TE (topographic error):** the share of stocks whose best and second-best neurons are not grid neighbours. It measures how well topology is preserved.

### 6. From neurons to clusters
49 neurons are too many groups to interpret. The project runs K-means on the 49 trained **weight vectors** for k = 3…10 and keeps the k with the best silhouette score. Each stock then inherits its BMU's cluster.

The **U-matrix** gives a visual check. It plots the mean distance from each neuron to its neighbours: bright "walls" mark cluster boundaries and dark "valleys" are cluster interiors.

### 7. SOM vs LVQ vs autoencoder
| | Learning | Rule |
|---|---|---|
| SOM | Unsupervised competitive | Winner and its neighbours move towards x |
| LVQ1 | Supervised competitive | Winner moves towards x if its label is right, away if wrong; no neighbourhood |
| Autoencoder | Self-supervised, backpropagation | Minimise ‖x − x̂‖² through a 3-unit bottleneck |

LVQ asks: "given only behaviour, can a competitive network guess the sector?" It is cross-validated against a majority-class baseline. Sectors with fewer than 3 stocks are merged into "Other" for LVQ only, because one prototype fitted to 1–2 stocks is not a meaningful sector prototype.

### 8. The autoencoder (Units 4–5)
The network is 7 → 16 → **3** → 16 → 7, with ReLU hidden layers, a linear output and MSE loss. Each Unit 4 concept is visible in the code:

- **He initialisation** suits ReLU layers.
- **Dropout** randomly switches off hidden units during training.
- **L2** penalises large weights.
- **Adam** is the optimiser.
- **Early stopping** watches the validation loss and restores the best weights.

The SOM can then be trained on the 3-D codes, and RQ1 compares ARI, NMI, QE and TE for SOM (raw features), K-means and autoencoder + SOM.

### 9. The RQ3 test
For 2,000 random portfolios of k stocks under each rule (one stock per SOM cluster vs. k different industries), the backend computes 2025 volatility, return, Sharpe ratio and maximum drawdown. It then compares the two distributions with a **Mann–Whitney U test**, a rank test that does not assume normality. The verdict sentence is generated only from those numbers.

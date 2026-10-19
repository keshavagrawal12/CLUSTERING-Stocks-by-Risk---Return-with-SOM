// TypeScript mirrors of the backend's Pydantic models (backend/app/api/schemas.py).

export type FeatureDict = Record<string, number | null>
export type Grid = number[][]

export interface Dropped { ticker: string; reason: string }

export interface TrainParams {
  rows: number
  cols: number
  sigma0: number
  eta0: number
  ordering_iters: number
  convergence_iters: number
  init: 'pca' | 'random'
  input_mode: 'raw' | 'autoencoder'
  seed: number
  autoencoder?: Record<string, number> | null
}

export interface Status {
  loaded: boolean
  n_stocks: number
  n_industries: number
  dropped: Dropped[]
  source: string | null
  load_error: string | null
  feature_check: Record<string, number | null>
  features: string[]
  feature_units: Record<string, string>
  periods: { clustering: [string, string]; test: [string, string] }
  autoencoder_available: boolean
  autoencoder_reason: string | null
  latest_run_id: string | null
  suggested_neurons: number | null
  defaults: TrainParams
}

export interface StockRow {
  ticker: string
  symbol: string
  company: string
  industry: string
  features: FeatureDict
  z: FeatureDict
}

export interface StocksResponse {
  stocks: StockRow[]
  industries: string[]
  features: string[]
  feature_units: Record<string, string>
}

export interface RunSummary {
  run_id: string
  status: string
  params: TrainParams
  created_at: number
  finished_at: number | null
  k: number | null
  qe: number | null
  te: number | null
  error: string | null
}

export interface Checkpoint {
  type: 'checkpoint'
  iteration: number
  total: number
  phase: 'ordering' | 'convergence'
  eta: number
  sigma: number
  qe: number
  te: number
  umatrix: Grid
  hits: Grid
  weights: number[][][]
}

export interface AEProgress {
  type: 'autoencoder'
  epochs_run: number
  best_epoch: number
  train_loss: number[]
  val_loss: number[]
}

export type StreamEvent =
  | Checkpoint
  | AEProgress
  | { type: 'status'; status: string; message: string }
  | { type: 'done'; run_id: string; k: number; qe: number; te: number }
  | { type: 'error'; message: string }

export interface MapStock {
  ticker: string
  symbol: string
  company: string
  industry: string
  row: number
  col: number
  cluster: number
}

export interface MapData {
  run_id: string
  rows: number
  cols: number
  input_mode: string
  hits: Grid
  umatrix: Grid
  component_planes: { feature: string; unit: string; values: Grid }[]
  planes_decoded: boolean
  neuron_clusters: Grid
  stocks: MapStock[]
  qe: number
  te: number
}

export interface ClusterProfile {
  cluster: number
  size: number
  members: string[]
  mean: FeatureDict | null
  median: FeatureDict | null
  industries: { industry: string; count: number }[]
  neurons: number
}

export interface ClustersData {
  run_id: string
  k: number
  silhouette: { k: number; score: number }[]
  stock_clusters: Record<string, number>
  neuron_clusters: Grid
  profiles: ClusterProfile[]
}

export interface StockDetail {
  ticker: string
  company: string
  industry: string
  symbol: string
  bmu: [number, number]
  cluster: number
  features: FeatureDict
  z: FeatureDict
  sector_median: FeatureDict
  sector_size: number
  cluster_mean: FeatureDict
  cluster_size: number
}

export interface LVQData {
  accuracy: number
  accuracy_std: number
  fold_accuracies: number[]
  baseline_accuracy: number
  baseline_class: string
  classes: string[]
  confusion: number[][]
  merged_into_other: string[]
  n_samples: number
  folds: number
}

export interface Variant {
  name: string
  key: string
  available: boolean
  k?: number | null
  ari?: number | null
  nmi?: number | null
  qe?: number | null
  te?: number | null
  reason?: string | null
  ae_reconstruction_mse?: number | null
}

export interface RQ1Data {
  run_id: string
  input_mode: string
  k: number
  ari: number
  nmi: number
  headline: string
  variants: Variant[]
  contingency: { clusters: number[]; industries: string[]; counts: number[][] }
  silhouette: { k: number; score: number }[]
  lvq: LVQData
}

export interface Misfit {
  rank: number
  ticker: string
  company: string
  industry: string
  sector_size: number
  bmu: [number, number]
  sector_centre: [number, number]
  grid_distance: number
  cluster: number
  sector_modal_cluster: number
  cluster_mismatch: boolean
  features: FeatureDict
  sector_median: FeatureDict
  z_diff: FeatureDict
  explanation: string
}

export interface RQ2Data {
  run_id: string
  misfits: Misfit[]
  n_cluster_mismatch: number
  n_ranked: number
  headline: string
}

export interface DistSummary {
  median: number
  q1: number
  q3: number
  mean: number
  min: number
  max: number
  whisker_low: number
  whisker_high: number
  sample: number[]
}

export interface RQ3Data {
  run_id: string
  k: number
  n_portfolios: number
  n_eligible: number
  excluded: Dropped[]
  period: [string, string]
  trading_days: number
  distributions: Record<'volatility' | 'annual_return' | 'sharpe' | 'max_drawdown', { cluster: DistSummary; sector: DistSummary }>
  median_vol_diff: number
  mann_whitney: { u: number; p_two_sided: number; p_cluster_lower: number; sharpe_p_two_sided: number; alpha: number }
  curves: {
    dates: string[]
    cluster: number[]
    sector: number[]
    nifty: (number | null)[]
    cluster_members: string[]
    sector_members: string[]
  }
  verdict: string
}

export interface StepData {
  ticker: string
  features: string[]
  x: number[]
  bmu: [number, number]
  distances: Grid
  neighbourhood: Grid
  weights_before: number[][][]
  weights_after: number[][][]
  eta: number
  sigma: number
}

// Thin typed wrapper around the FastAPI endpoints. All requests go to /api (proxied by Vite).
import type {
  ClustersData, MapData, RQ1Data, RQ2Data, RQ3Data, RunSummary, Status, StepData,
  StockDetail, StocksResponse, StreamEvent, TrainParams,
} from './types'

export class ApiError extends Error {
  status: number
  code?: string
  constructor(status: number, message: string, code?: string) {
    super(message)
    this.status = status
    this.code = code
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`/api${path}`, init)
  } catch {
    throw new ApiError(0, 'Cannot reach the backend. Is it running on port 8000?', 'offline')
  }
  // The Vite proxy answers 502/504 when the backend is down or still starting up.
  if (res.status === 502 || res.status === 504) {
    throw new ApiError(0, 'Cannot reach the backend on port 8000 (it may still be starting, since loading TensorFlow takes a few seconds).', 'offline')
  }
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`
    let code: string | undefined
    try {
      const body = await res.json()
      const d = body.detail
      if (typeof d === 'string') message = d
      else if (d?.message) { message = d.message; code = d.code }
      else if (Array.isArray(d)) message = d.map((e: { msg: string }) => e.msg).join('; ')
    } catch { /* non-JSON error body */ }
    throw new ApiError(res.status, message, code)
  }
  return res.json() as Promise<T>
}

const post = <T,>(path: string, body: unknown) =>
  request<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export const api = {
  status: () => request<Status>('/status'),
  reload: () => request<Status>('/data/reload', { method: 'POST' }),
  upload: (files: { features: File; prices: File; volumes: File }) => {
    const fd = new FormData()
    fd.append('features', files.features)
    fd.append('prices', files.prices)
    fd.append('volumes', files.volumes)
    return request<Status & { message: string }>('/data/upload', { method: 'POST', body: fd })
  },
  stocks: () => request<StocksResponse>('/stocks'),
  runs: () => request<{ runs: RunSummary[]; latest_run_id: string | null }>('/runs'),
  train: (p: TrainParams) => post<{ run_id: string; status: string }>('/som/train', p),
  map: (id: string) => request<MapData>(`/som/${id}/map`),
  clusters: (id: string) => request<ClustersData>(`/som/${id}/clusters`),
  stock: (id: string, ticker: string) => request<StockDetail>(`/som/${id}/stock/${encodeURIComponent(ticker)}`),
  rq1: (id: string) => request<RQ1Data>(`/som/${id}/rq1`),
  rq2: (id: string) => request<RQ2Data>(`/som/${id}/rq2`),
  rq3: (id: string) => request<RQ3Data>(`/som/${id}/rq3`),
  step: (body: {
    ticker: string; weights?: number[][][] | null; rows: number; cols: number
    eta: number; sigma: number; init: 'pca' | 'random'; seed: number
  }) => post<StepData>('/som/step', body),
}

/** Subscribe to the SSE training stream. Returns a function that closes it. */
export function streamTraining(runId: string, onEvent: (e: StreamEvent) => void, onFail: (msg: string) => void) {
  const es = new EventSource(`/api/som/train/${runId}/stream`)
  let finished = false
  const handle = (ev: MessageEvent) => {
    const data = JSON.parse(ev.data) as StreamEvent
    onEvent(data)
    if (data.type === 'done' || data.type === 'error') {
      finished = true
      es.close()
    }
  }
  for (const t of ['checkpoint', 'autoencoder', 'status', 'done', 'error']) es.addEventListener(t, handle)
  es.onerror = () => {
    if (!finished) onFail('Lost connection to the training stream.')
    es.close()
  }
  return () => es.close()
}

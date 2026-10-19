/* eslint-disable react-refresh/only-export-components */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { api } from '../api/client'
import type { Status, StocksResponse } from '../api/types'
import { buildSectorPalette, type SectorPalette, type Theme } from './palette'

interface AppState {
  status: Status | null
  statusError: string | null
  stocks: StocksResponse | null
  runId: string | null
  setRunId: (id: string | null) => void
  refresh: () => Promise<void>
  theme: Theme
  toggleTheme: () => void
  palette: SectorPalette
  industryOf: (ticker: string) => string
}

const Ctx = createContext<AppState | null>(null)

function readTheme(): Theme {
  try {
    const t = localStorage.getItem('theme')
    if (t === 'light' || t === 'dark') return t
  } catch { /* storage blocked */ }
  return 'dark'
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<Status | null>(null)
  const [statusError, setStatusError] = useState<string | null>(null)
  const [stocks, setStocks] = useState<StocksResponse | null>(null)
  const [runId, setRunIdState] = useState<string | null>(null)
  const [theme, setTheme] = useState<Theme>(readTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { localStorage.setItem('theme', theme) } catch { /* ignore */ }
  }, [theme])

  const refresh = useCallback(async () => {
    try {
      const s = await api.status()
      setStatus(s)
      setStatusError(null)
      // Keep the selected run only if the backend still has it (runs live in memory and
      // disappear when the server restarts or new data is loaded).
      const known = s.loaded ? new Set((await api.runs()).runs.map((r) => r.run_id)) : new Set<string>()
      setRunIdState((cur) => (s.loaded ? (cur && known.has(cur) ? cur : s.latest_run_id) : null))
      setStocks(s.loaded ? await api.stocks() : null)
    } catch (e) {
      setStatusError((e as Error).message)
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  // While the backend is unreachable (e.g. still starting), retry every 3 seconds.
  useEffect(() => {
    if (!statusError) return
    const t = setTimeout(() => { void refresh() }, 3000)
    return () => clearTimeout(t)
  }, [statusError, refresh])

  const palette = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const s of stocks?.stocks ?? []) counts[s.industry] = (counts[s.industry] ?? 0) + 1
    return buildSectorPalette(counts, theme)
  }, [stocks, theme])

  const industryOf = useMemo(() => {
    const m = new Map((stocks?.stocks ?? []).map((s) => [s.ticker, s.industry]))
    return (t: string) => m.get(t) ?? 'Unknown'
  }, [stocks])

  const value: AppState = {
    status, statusError, stocks, runId,
    setRunId: setRunIdState, refresh, theme,
    toggleTheme: () => setTheme((t) => (t === 'dark' ? 'light' : 'dark')),
    palette, industryOf,
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useApp() {
  const v = useContext(Ctx)
  if (!v) throw new Error('useApp outside provider')
  return v
}

/** Small data-fetching hook: re-runs when `key` changes; null key = don't fetch. */
export function useFetch<T>(key: string | null, fn: () => Promise<T>) {
  const [state, setState] = useState<{ key: string | null; data: T | null; error: string | null }>(
    { key: null, data: null, error: null })
  useEffect(() => {
    if (!key) return
    let alive = true
    fn().then(
      (data) => alive && setState({ key, data, error: null }),
      (e: Error) => alive && setState({ key, data: null, error: e.message }),
    )
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  const fresh = state.key === key
  return { data: fresh ? state.data : null, error: fresh ? state.error : null, loading: !!key && !fresh }
}

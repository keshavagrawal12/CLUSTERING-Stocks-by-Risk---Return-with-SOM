import { useEffect, useState } from 'react'
import { api } from './api/client'
import type { RunSummary } from './api/types'
import { TooltipProvider } from './components/ui'
import { PAGES, navigate, useRoute, type PageId } from './lib/router'
import { AppProvider, useApp } from './lib/store'
import Overview from './pages/Overview'
import DataPage from './pages/Data'
import TrainPage from './pages/Train'
import MapPage from './pages/Map'
import ClustersPage from './pages/Clusters'
import MisfitsPage from './pages/Misfits'
import PortfolioPage from './pages/Portfolio'
import HowPage from './pages/How'

const VIEWS: Record<PageId, () => React.JSX.Element> = {
  overview: Overview, data: DataPage, train: TrainPage, map: MapPage,
  clusters: ClustersPage, misfits: MisfitsPage, portfolio: PortfolioPage, how: HowPage,
}

function RunPicker() {
  const { runId, setRunId, status } = useApp()
  const [runs, setRuns] = useState<RunSummary[]>([])
  useEffect(() => {
    if (!status?.loaded) { setRuns([]); return }
    api.runs().then((r) => setRuns(r.runs.filter((x) => x.status === 'done'))).catch(() => setRuns([]))
  }, [status, runId])
  if (!runs.length) return <span className="text-[12px] text-ink-3">No run yet</span>
  return (
    <label className="flex items-center gap-2 text-[12px] text-ink-3">
      Run
      <select value={runId ?? ''} onChange={(e) => setRunId(e.target.value)}
        className="num rounded-md border border-line bg-panel-2 px-2 py-1 text-[12px] text-ink outline-none focus:border-accent">
        {runs.map((r) => (
          <option key={r.run_id} value={r.run_id}>
            {r.run_id.slice(0, 6)} · {r.params.rows}×{r.params.cols} · {r.params.input_mode === 'raw' ? 'raw' : 'AE'} · k={r.k}
          </option>
        ))}
      </select>
    </label>
  )
}

function Shell() {
  const page = useRoute()
  const { status, theme, toggleTheme } = useApp()
  const View = VIEWS[page]
  const groups = [...new Set(PAGES.map((p) => p.group))]
  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-[232px] shrink-0 flex-col border-r border-line bg-panel/60 px-4 py-6 md:flex">
        <button onClick={() => navigate('overview')} className="mb-8 px-2 text-left">
          <div className="text-[11px] font-medium uppercase tracking-[0.16em] text-accent">BDA404 · ANN</div>
          <div className="mt-1 text-[15px] font-semibold leading-snug text-ink">Beyond Sector Labels</div>
          <div className="mt-0.5 text-[12px] text-ink-3">NIFTY 100 · Kohonen SOM</div>
        </button>
        <nav className="flex-1 space-y-5">
          {groups.map((g) => (
            <div key={g}>
              <div className="mb-1.5 px-2 text-[10.5px] font-medium uppercase tracking-[0.14em] text-ink-3">{g}</div>
              {PAGES.filter((p) => p.group === g).map((p) => (
                <a key={p.id} href={`#/${p.id}`}
                  className={`flex items-center justify-between rounded-md px-2 py-1.5 text-[13.5px] transition-colors ${page === p.id ? 'bg-accent-soft text-ink' : 'text-ink-2 hover:bg-panel-2 hover:text-ink'}`}>
                  <span className="flex items-center gap-2">
                    <span className={`h-1.5 w-1.5 rounded-full ${page === p.id ? 'bg-accent' : 'bg-transparent'}`} />
                    {p.label}
                  </span>
                  {'tag' in p && <span className="num text-[10.5px] text-ink-3">{p.tag}</span>}
                </a>
              ))}
            </div>
          ))}
        </nav>
        <div className="space-y-1 px-2 text-[11.5px] leading-relaxed text-ink-3">
          <div className="flex items-center gap-2">
            <span className={`h-1.5 w-1.5 rounded-full ${status?.loaded ? 'bg-good' : 'bg-critical'}`} />
            {status?.loaded ? <span className="num">{status.n_stocks} stocks loaded</span> : 'No data'}
          </div>
          <div className="num">Fit 2022–24 · Test 2025</div>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="sticky top-0 z-20 flex items-center justify-between gap-4 border-b border-line bg-bg/85 px-6 py-2.5 backdrop-blur md:px-10">
          <select className="rounded-md border border-line bg-panel px-2 py-1 text-[13px] text-ink md:hidden" value={page}
            onChange={(e) => navigate(e.target.value as PageId)}>
            {PAGES.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
          </select>
          <div className="hidden text-[12px] text-ink-3 md:block">
            {PAGES.find((p) => p.id === page)?.group} / <span className="text-ink-2">{PAGES.find((p) => p.id === page)?.label}</span>
          </div>
          <div className="flex items-center gap-4">
            <RunPicker />
            <button onClick={toggleTheme} aria-label="Toggle colour theme"
              className="rounded-md border border-line px-2.5 py-1 text-[12px] text-ink-2 hover:text-ink">
              {theme === 'dark' ? 'Light' : 'Dark'} theme
            </button>
          </div>
        </div>
        <main className="mx-auto w-full max-w-[1320px] flex-1 px-6 py-9 md:px-10">
          <View />
        </main>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <AppProvider>
      <TooltipProvider>
        <Shell />
      </TooltipProvider>
    </AppProvider>
  )
}

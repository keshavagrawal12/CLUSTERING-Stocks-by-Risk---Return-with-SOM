import * as d3 from 'd3'
import { useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api/client'
import type { StockRow } from '../api/types'
import { useChartStyle } from '../components/charts'
import { Badge, Button, Card, EmptyState, ErrorBox, Loading, PageHeader, SectorDot, Spinner } from '../components/ui'
import { FEATURE_LABEL, FEATURE_UNIT_LABEL, axisValue, fmtFeature, shortTicker } from '../lib/format'
import { OTHER } from '../lib/palette'
import { useApp } from '../lib/store'

function FileField({ label, file, onFile }: { label: string; file: File | null; onFile: (f: File | null) => void }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 rounded-lg border border-dashed border-line-strong px-3 py-2.5 hover:border-ink-3">
      <span className="num text-[12.5px] text-ink-2">{label}</span>
      <span className="truncate text-[12px] text-ink-3">{file ? file.name : 'Choose file…'}</span>
      <input type="file" accept=".csv,text/csv" className="hidden" onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
    </label>
  )
}

function Upload() {
  const { refresh, setRunId } = useApp()
  const [files, setFiles] = useState<{ features: File | null; prices: File | null; volumes: File | null }>({ features: null, prices: null, volumes: null })
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null)
  const ready = files.features && files.prices && files.volumes

  const submit = async () => {
    if (!ready) return
    setBusy(true); setMsg(null)
    try {
      const r = await api.upload({ features: files.features!, prices: files.prices!, volumes: files.volumes! })
      setRunId(null)
      await refresh()
      setMsg({ ok: true, text: `${r.message}. Previous runs were cleared.` })
    } catch (e) { setMsg({ ok: false, text: (e as Error).message }) } finally { setBusy(false) }
  }
  const reload = async () => {
    setBusy(true); setMsg(null)
    try {
      const s = await api.reload()
      setRunId(null)
      await refresh()
      setMsg(s.loaded ? { ok: true, text: `Loaded ${s.n_stocks} stocks from backend/data/raw/` } : { ok: false, text: s.load_error ?? 'No files found' })
    } catch (e) { setMsg({ ok: false, text: (e as Error).message }) } finally { setBusy(false) }
  }

  return (
    <Card title="Load data" subtitle="Upload the three CSVs from the Colab notebook, or re-read backend/data/raw/">
      <div className="space-y-2">
        <FileField label="nifty100_features.csv" file={files.features} onFile={(f) => setFiles((s) => ({ ...s, features: f }))} />
        <FileField label="nifty100_prices.csv" file={files.prices} onFile={(f) => setFiles((s) => ({ ...s, prices: f }))} />
        <FileField label="nifty100_volumes.csv" file={files.volumes} onFile={(f) => setFiles((s) => ({ ...s, volumes: f }))} />
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <Button onClick={submit} disabled={!ready || busy}>{busy ? <Spinner /> : null}Upload &amp; compute features</Button>
        <Button variant="ghost" onClick={reload} disabled={busy}>Reload from folder</Button>
      </div>
      {msg && <div className={`mt-3 text-[12.5px] ${msg.ok ? 'text-good' : 'text-critical'}`}>{msg.text}</div>}
      <p className="mt-4 text-[12px] leading-relaxed text-ink-3">
        The backend recomputes all seven features from the price file using only 2022-01-01 to 2024-12-31, so 2025 prices can never
        leak into the model. Or rebuild everything with <span className="num">python -m scripts.fetch_data</span>.
      </p>
    </Card>
  )
}

function Histogram({ feature, stocks, industry }: { feature: string; stocks: StockRow[]; industry: string }) {
  const { tick, tooltip, axisLabel, ink } = useChartStyle()
  const { palette, theme } = useApp()
  const data = useMemo(() => {
    const vals = stocks.map((s) => ({ v: axisValue(feature, s.features[feature] ?? NaN), sel: s.industry === industry }))
      .filter((d) => Number.isFinite(d.v))
    const bins = d3.bin<{ v: number; sel: boolean }, number>().value((d) => d.v).thresholds(12)(vals)
    return bins.map((b) => ({
      x: (b.x0! + b.x1!) / 2, x0: b.x0!, x1: b.x1!,
      selected: b.filter((d) => d.sel).length, rest: b.filter((d) => !d.sel).length,
    }))
  }, [stocks, feature, industry])
  const fmt = (v: number) => (feature === 'Log_Traded_Value' ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1))
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-[12.5px] font-medium text-ink">{FEATURE_LABEL[feature]}</span>
        <span className="text-[11px] text-ink-3">{FEATURE_UNIT_LABEL[feature]}</span>
      </div>
      <ResponsiveContainer width="100%" height={150}>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 14, left: 2 }} barCategoryGap={1}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="x" tick={tick} tickFormatter={fmt} interval="preserveStartEnd" label={axisLabel(FEATURE_UNIT_LABEL[feature])} />
          <YAxis tick={tick} allowDecimals={false} width={46} label={axisLabel("stocks", -90)} />
          <Tooltip {...tooltip} cursor={{ fill: ink.grid }}
            labelFormatter={(_, p) => { const d = p?.[0]?.payload; return d ? `${fmt(d.x0)} – ${fmt(d.x1)} ${FEATURE_UNIT_LABEL[feature]}` : '' }} />
          {industry && <Bar dataKey="selected" stackId="a" name={industry} fill={palette.color(industry)} radius={[0, 0, 0, 0]} />}
          <Bar dataKey="rest" stackId="a" name={industry ? 'Other stocks' : 'Stocks'} fill={industry ? OTHER[theme] : palette.color(palette.coloured[0] ?? '') } radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

type SortKey = 'ticker' | 'industry' | string

export default function DataPage() {
  const { status, stocks, statusError } = useApp()
  const [q, setQ] = useState('')
  const [industry, setIndustry] = useState('')
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: 'Sharpe_Ratio', dir: -1 })

  const rows = useMemo(() => {
    if (!stocks) return []
    const needle = q.trim().toLowerCase()
    const r = stocks.stocks.filter((s) =>
      (!industry || s.industry === industry) &&
      (!needle || s.ticker.toLowerCase().includes(needle) || s.company.toLowerCase().includes(needle)))
    const val = (s: StockRow) => (sort.key === 'ticker' ? s.symbol : sort.key === 'industry' ? s.industry : s.features[sort.key] ?? -Infinity)
    return [...r].sort((a, b) => { const va = val(a), vb = val(b); return (va < vb ? -1 : va > vb ? 1 : 0) * sort.dir })
  }, [stocks, q, industry, sort])

  const head = (key: SortKey, label: string, right = true) => (
    <th className={`cursor-pointer whitespace-nowrap px-3 py-2 font-medium hover:text-ink ${right ? 'text-right' : 'text-left'}`}
      onClick={() => setSort((s) => ({ key, dir: s.key === key ? (s.dir === 1 ? -1 : 1) : -1 }))}>
      {label}{sort.key === key ? (sort.dir === 1 ? ' ↑' : ' ↓') : ''}
    </th>
  )

  return (
    <div>
      <PageHeader eyebrow="Project" title="Data">
        One row per stock: seven behaviour features measured over the clustering period
        (<span className="num">{status?.periods.clustering.join(' → ') ?? '2022-01-01 → 2024-12-31'}</span>). These are the inputs to every neural network in the project.
      </PageHeader>

      <div className="mb-6 grid gap-4 lg:grid-cols-[1fr_1fr]">
        <Upload />
        <Card title="Load report" subtitle="Cleaning and sanity checks from the backend">
          {statusError ? <ErrorBox message={statusError} /> : !status ? <Loading /> : !status.loaded ? (
            <div className="text-[13px] text-ink-3">Nothing loaded yet.{status.load_error && <div className="mt-2 text-critical">{status.load_error}</div>}</div>
          ) : (
            <div className="space-y-4 text-[13px]">
              <div>
                <div className="mb-1.5 text-[12px] font-medium text-ink-2">Dropped (&gt; 10% of trading days missing)</div>
                {status.dropped.length === 0 ? <span className="text-ink-3">None. Every stock had enough history.</span> : (
                  <ul className="space-y-1">{status.dropped.map((d) => (
                    <li key={d.ticker} className="flex justify-between gap-3"><span className="num text-ink">{d.ticker}</span><span className="text-ink-3">{d.reason}</span></li>
                  ))}</ul>
                )}
              </div>
              <div>
                <div className="mb-1.5 text-[12px] font-medium text-ink-2">Recomputed vs CSV features (max absolute difference)</div>
                <div className="grid grid-cols-2 gap-x-6 gap-y-1">
                  {Object.entries(status.feature_check).map(([f, v]) => (
                    <div key={f} className="flex justify-between"><span className="text-ink-3">{FEATURE_LABEL[f]}</span>
                      <span className={`num ${v != null && v > 1e-3 ? 'text-critical' : 'text-ink-2'}`}>{v == null ? '—' : v.toExponential(1)}</span></div>
                  ))}
                </div>
                <p className="mt-2 text-[11.5px] text-ink-3">Tiny values mean the backend's formulas reproduce your notebook's numbers.</p>
              </div>
            </div>
          )}
        </Card>
      </div>

      {!status?.loaded ? (
        status && <EmptyState title="No data loaded">Upload the three CSVs above (or place them in <span className="num">backend/data/raw/</span> and press Reload).</EmptyState>
      ) : !stocks ? <Loading tall /> : (
        <>
          <Card title="Feature distributions" subtitle={industry ? `${industry} highlighted against all other stocks` : 'All stocks · pick an industry below to highlight it'} className="mb-6">
            <div className="grid gap-x-6 gap-y-5 sm:grid-cols-2 lg:grid-cols-4">
              {stocks.features.map((f) => <Histogram key={f} feature={f} stocks={stocks.stocks} industry={industry} />)}
            </div>
          </Card>

          <Card pad={false} title={<span>Stocks <span className="num font-normal text-ink-3">· {rows.length} of {stocks.stocks.length}</span></span>}
            right={
              <div className="flex flex-wrap gap-2">
                <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search ticker or company"
                  className="w-52 rounded-md border border-line bg-panel-2 px-2.5 py-1.5 text-[12.5px] text-ink outline-none placeholder:text-ink-3 focus:border-accent" />
                <select value={industry} onChange={(e) => setIndustry(e.target.value)}
                  className="rounded-md border border-line bg-panel-2 px-2 py-1.5 text-[12.5px] text-ink outline-none focus:border-accent">
                  <option value="">All industries</option>
                  {stocks.industries.map((i) => <option key={i} value={i}>{i}</option>)}
                </select>
              </div>
            }>
            <div className="scroll-thin max-h-[560px] overflow-auto">
              <table className="w-full text-[12.5px]">
                <thead className="sticky top-0 z-10 bg-panel text-[11.5px] text-ink-3">
                  <tr className="border-b border-line">
                    {head('ticker', 'Ticker', false)}
                    {head('industry', 'Industry', false)}
                    {stocks.features.map((f) => head(f, FEATURE_LABEL[f]))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((s) => (
                    <tr key={s.ticker} className="border-b border-line/60 hover:bg-panel-2">
                      <td className="px-3 py-1.5"><div className="num font-medium text-ink">{shortTicker(s.ticker)}</div><div className="max-w-[200px] truncate text-[11px] text-ink-3">{s.company}</div></td>
                      <td className="px-3 py-1.5"><span className="flex items-center gap-1.5 text-ink-2"><SectorDot industry={s.industry} />{s.industry}</span></td>
                      {stocks.features.map((f) => <td key={f} className="num px-3 py-1.5 text-right text-ink-2">{fmtFeature(f, s.features[f])}</td>)}
                    </tr>
                  ))}
                  {rows.length === 0 && <tr><td colSpan={9} className="px-3 py-10 text-center text-ink-3">No stocks match. <Badge>clear the filter</Badge></td></tr>}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  )
}

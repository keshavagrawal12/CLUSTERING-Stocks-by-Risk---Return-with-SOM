import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api/client'
import type { RQ3Data } from '../api/types'
import { BoxPlot, LegendRow, useChartStyle } from '../components/charts'
import { Badge, Card, ErrorBox, Loading, PageHeader, Requires, Stat } from '../components/ui'
import { num, pct, pValue, shortTicker } from '../lib/format'
import { ruleColors } from '../lib/palette'
import { useApp, useFetch } from '../lib/store'

export default function PortfolioPage() {
  return <Requires run>{(runId) => <PortfolioView runId={runId} />}</Requires>
}

const METRICS: { key: keyof RQ3Data['distributions']; label: string; axis: string; fmt: (v: number) => string; better: 'lower' | 'higher' }[] = [
  { key: 'volatility', label: 'Annualised volatility', axis: 'volatility, % p.a.', fmt: (v) => `${(v * 100).toFixed(0)}%`, better: 'lower' },
  { key: 'sharpe', label: 'Sharpe ratio', axis: 'Sharpe ratio (rf = 6.5%)', fmt: (v) => v.toFixed(1), better: 'higher' },
  { key: 'annual_return', label: 'Annual return', axis: 'return, % (2025)', fmt: (v) => `${(v * 100).toFixed(0)}%`, better: 'higher' },
  { key: 'max_drawdown', label: 'Max drawdown', axis: 'max drawdown, %', fmt: (v) => `${(v * 100).toFixed(0)}%`, better: 'higher' },
]

function PortfolioView({ runId }: { runId: string }) {
  const { data, error } = useFetch(`rq3-${runId}`, () => api.rq3(runId))
  const { theme } = useApp()
  if (error) return <ErrorBox message={error} />
  if (!data) return <Loading tall label="Running 2025 Monte Carlo…" />
  const c = ruleColors(theme)
  const mw = data.mann_whitney
  const sig = mw.p_two_sided < mw.alpha

  return (
    <div>
      <PageHeader eyebrow="Findings · RQ3 · out-of-sample" title="Does cluster diversification lower risk in 2025?">
        Clusters were learned on 2022–2024. Here we build <span className="num">{data.n_portfolios.toLocaleString()}</span> random equal-weight portfolios of{' '}
        <span className="num">k = {data.k}</span> stocks under each rule and measure them on {data.period[0]} → {data.period[1]} only
        (<span className="num">{data.trading_days}</span> trading days, <span className="num">{data.n_eligible}</span> eligible stocks).
      </PageHeader>

      <div className="mb-5 rounded-xl border border-accent/40 bg-accent-soft px-5 py-4">
        <div className="text-[11px] font-medium uppercase tracking-[0.12em] text-accent">Verdict</div>
        <p className="mt-1.5 text-[16px] leading-snug text-ink">{data.verdict}</p>
      </div>

      <div className="mb-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card><Stat label="Median vol · clusters" value={pct(data.distributions.volatility.cluster.median)} sub="one stock from each SOM cluster" /></Card>
        <Card><Stat label="Median vol · sectors" value={pct(data.distributions.volatility.sector.median)} sub="k different industries" /></Card>
        <Card><Stat label="Difference" value={`${data.median_vol_diff >= 0 ? '+' : ''}${(data.median_vol_diff * 100).toFixed(2)} pp`}
          sub={data.median_vol_diff < 0 ? 'clusters less volatile' : 'clusters more volatile'} tone="accent" /></Card>
        <Card><Stat label="Mann–Whitney p" value={pValue(mw.p_two_sided)}
          sub={<>{sig ? <Badge tone="accent">significant at α = {mw.alpha}</Badge> : <Badge>not significant at α = {mw.alpha}</Badge>}</>} /></Card>
      </div>

      <Card className="mb-5" title="Distribution of 2025 outcomes" subtitle="Box = quartiles, thick line = median, whiskers = 1.5×IQR; dots are a random sample of individual portfolios. Hover a box for exact values."
        right={<LegendRow items={[{ label: 'Cluster-diversified', color: c.cluster }, { label: 'Sector-diversified', color: c.sector }]} />}>
        <div className="grid gap-x-10 gap-y-6 md:grid-cols-2">
          {METRICS.map((m) => (
            <div key={m.key}>
              <div className="mb-1 flex items-baseline justify-between">
                <span className="text-[12.5px] font-medium text-ink">{m.label}</span>
                <span className="text-[11px] text-ink-3">{m.better} is better</span>
              </div>
              <BoxPlot axisLabel={m.axis} format={m.fmt} groups={[
                { label: 'Clusters', color: c.cluster, d: data.distributions[m.key].cluster },
                { label: 'Sectors', color: c.sector, d: data.distributions[m.key].sector },
              ]} />
              <div className="num mt-1 flex justify-between text-[11.5px] text-ink-3">
                <span>median {m.key === 'sharpe' ? num(data.distributions[m.key].cluster.median) : pct(data.distributions[m.key].cluster.median)}</span>
                <span>median {m.key === 'sharpe' ? num(data.distributions[m.key].sector.median) : pct(data.distributions[m.key].sector.median)}</span>
              </div>
            </div>
          ))}
        </div>
        <p className="mt-4 text-[12px] text-ink-3">
          Sharpe ratio distributions: Mann–Whitney p = <span className="num">{pValue(mw.sharpe_p_two_sided)}</span>.
          One-sided test that cluster portfolios are <em>less</em> volatile: p = <span className="num">{pValue(mw.p_cluster_lower)}</span>.
        </p>
      </Card>

      <Curves data={data} />

      {data.excluded.length > 0 && (
        <p className="mt-4 text-[12px] text-ink-3">Excluded from RQ3 (insufficient 2025 data): {data.excluded.map((e) => `${shortTicker(e.ticker)} (${e.reason})`).join(', ')}.</p>
      )}
    </div>
  )
}

function Curves({ data }: { data: RQ3Data }) {
  const { tick, tooltip, axisLabel, ink } = useChartStyle()
  const { theme } = useApp()
  const c = ruleColors(theme)
  const rows = data.curves.dates.map((d, i) => ({ date: d, cluster: data.curves.cluster[i] * 100, sector: data.curves.sector[i] * 100, nifty: (data.curves.nifty[i] ?? NaN) * 100 }))
  const monthTick = (d: string) => new Date(d).toLocaleString('en-GB', { month: 'short' })
  return (
    <Card title="Median portfolio of each rule vs NIFTY 50" subtitle="Cumulative 2025 return of the portfolio with the median volatility under each rule"
      right={<LegendRow items={[{ label: 'Cluster-diversified', color: c.cluster }, { label: 'Sector-diversified', color: c.sector }, { label: 'NIFTY 50', color: c.nifty, dashed: true }]} />}>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 16, left: 8 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="date" tick={tick} tickFormatter={monthTick} minTickGap={30} label={axisLabel('2025')} />
          <YAxis tick={tick} tickFormatter={(v) => `${v.toFixed(0)}%`} width={52} label={axisLabel('cumulative return, %', -90)} />
          <ReferenceLine y={0} stroke={ink.text3} />
          <Tooltip {...tooltip} labelFormatter={(d) => new Date(String(d)).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}
            formatter={(v, n) => [`${Number(v).toFixed(2)}%`, n === 'cluster' ? 'Cluster-diversified' : n === 'sector' ? 'Sector-diversified' : 'NIFTY 50']} />
          <Line dataKey="nifty" stroke={c.nifty} strokeWidth={2} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
          <Line dataKey="sector" stroke={c.sector} strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line dataKey="cluster" stroke={c.cluster} strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-3 grid gap-3 text-[12px] sm:grid-cols-2">
        <div><span className="text-ink-3">Cluster portfolio: </span><span className="num text-ink-2">{data.curves.cluster_members.map(shortTicker).join(', ')}</span></div>
        <div><span className="text-ink-3">Sector portfolio: </span><span className="num text-ink-2">{data.curves.sector_members.map(shortTicker).join(', ')}</span></div>
      </div>
    </Card>
  )
}

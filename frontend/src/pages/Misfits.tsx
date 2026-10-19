import { useMemo, useState } from 'react'
import { api } from '../api/client'
import type { Misfit } from '../api/types'
import { ZBars } from '../components/charts'
import { Badge, Button, Card, ErrorBox, Loading, PageHeader, Requires, SectorDot, Segmented } from '../components/ui'
import { FEATURE_LABEL, fmtFeature, shortTicker } from '../lib/format'
import { useApp, useFetch } from '../lib/store'

export default function MisfitsPage() {
  return <Requires run>{(runId) => <MisfitsView runId={runId} />}</Requires>
}

function MisfitsView({ runId }: { runId: string }) {
  const { data, error } = useFetch(`rq2-${runId}`, () => api.rq2(runId))
  const [onlyMismatch, setOnlyMismatch] = useState<'all' | 'mismatch'>('all')
  const [industry, setIndustry] = useState('')
  const [limit, setLimit] = useState(20)

  const rows = useMemo(() => (data?.misfits ?? []).filter((m) =>
    (onlyMismatch === 'all' || m.cluster_mismatch) && (!industry || m.industry === industry)), [data, onlyMismatch, industry])
  const industries = useMemo(() => [...new Set((data?.misfits ?? []).map((m) => m.industry))].sort(), [data])

  if (error) return <ErrorBox message={error} />
  if (!data) return <Loading tall />
  const maxDist = Math.max(...data.misfits.map((m) => m.grid_distance), 1)

  return (
    <div>
      <PageHeader eyebrow="Findings · RQ2" title="Stocks that don't behave like their sector">
        <span className="text-ink">{data.headline}.</span> The score is the grid distance between a stock's neuron and the centre of
        the <em>other</em> stocks in its sector. The SOM preserves topology, so far away on the map means different behaviour.
        Stocks are also flagged when their cluster differs from their sector's most common cluster.
      </PageHeader>

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <Segmented value={onlyMismatch} onChange={setOnlyMismatch}
          options={[{ value: 'all', label: `All ranked (${data.n_ranked})` }, { value: 'mismatch', label: `Cluster mismatch (${data.n_cluster_mismatch})` }]} />
        <select value={industry} onChange={(e) => setIndustry(e.target.value)}
          className="rounded-md border border-line bg-panel-2 px-2 py-1.5 text-[12.5px] text-ink outline-none focus:border-accent">
          <option value="">All industries</option>
          {industries.map((i) => <option key={i} value={i}>{i}</option>)}
        </select>
        <span className="text-[12px] text-ink-3">Single-stock sectors are excluded (no sector centre to compare against).</span>
      </div>

      <div className="space-y-3">
        {rows.slice(0, limit).map((m) => <MisfitRow key={m.ticker} m={m} maxDist={maxDist} />)}
        {rows.length === 0 && <Card><p className="text-[13px] text-ink-3">No stocks match this filter.</p></Card>}
      </div>
      {rows.length > limit && (
        <div className="mt-4 flex justify-center"><Button variant="ghost" onClick={() => setLimit((l) => l + 20)}>Show 20 more ({rows.length - limit} left)</Button></div>
      )}
    </div>
  )
}

function MisfitRow({ m, maxDist }: { m: Misfit; maxDist: number }) {
  const { stocks } = useApp()
  const features = stocks?.features ?? Object.keys(m.features)
  return (
    <div className="grid gap-4 rounded-xl border border-line bg-panel p-4 md:grid-cols-[48px_1.1fr_1fr_1.15fr] md:items-center">
      <div className="num text-[20px] font-medium text-ink-3">{String(m.rank).padStart(2, '0')}</div>
      <div>
        <div className="flex flex-wrap items-baseline gap-2">
          <span className="num text-[15px] font-semibold text-ink">{shortTicker(m.ticker)}</span>
          <span className="truncate text-[12px] text-ink-3">{m.company}</span>
        </div>
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          <Badge><SectorDot industry={m.industry} />{m.industry} <span className="num text-ink-3">n={m.sector_size}</span></Badge>
          {m.cluster_mismatch
            ? <Badge tone="accent">C{m.cluster}, sector usually C{m.sector_modal_cluster}</Badge>
            : <Badge>C{m.cluster} (same as sector)</Badge>}
        </div>
        <p className="mt-2 text-[13px] leading-snug text-ink-2">{m.explanation}.</p>
        <div className="mt-2 flex items-center gap-2 text-[11.5px] text-ink-3">
          <span>grid distance</span>
          <div className="h-1.5 w-28 rounded-full bg-panel-2"><div className="h-full rounded-full bg-accent" style={{ width: `${(m.grid_distance / maxDist) * 100}%` }} /></div>
          <span className="num text-ink-2">{m.grid_distance.toFixed(2)} cells</span>
        </div>
      </div>
      <table className="w-full text-[11.5px]">
        <thead className="text-ink-3"><tr><th className="pb-1 text-left font-medium">Feature</th><th className="pb-1 text-right font-medium">Stock</th><th className="pb-1 text-right font-medium">Sector med.</th></tr></thead>
        <tbody>
          {features.map((f) => (
            <tr key={f}>
              <td className="py-0.5 text-ink-3">{FEATURE_LABEL[f]}</td>
              <td className="num py-0.5 text-right text-ink">{fmtFeature(f, m.features[f])}</td>
              <td className="num py-0.5 text-right text-ink-3">{fmtFeature(f, m.sector_median[f])}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div>
        <div className="mb-0.5 text-[11px] text-ink-3">vs sector median, in std-devs (blue below · red above)</div>
        <ZBars z={m.z_diff} features={features} reference={m.sector_median} refLabel="Sector median" compact />
      </div>
    </div>
  )
}

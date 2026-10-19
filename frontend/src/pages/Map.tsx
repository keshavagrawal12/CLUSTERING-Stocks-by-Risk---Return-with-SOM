import * as d3 from 'd3'
import { useMemo, useState } from 'react'
import { api } from '../api/client'
import type { MapData, MapStock } from '../api/types'
import { ZBars } from '../components/charts'
import { SomGrid } from '../components/SomGrid'
import {
  Badge, Card, ColorRamp, ErrorBox, Loading, PageHeader, Requires, SectorDot, SectorLegend, Segmented, TipRow, useTooltip,
} from '../components/ui'
import { FEATURE_LABEL, FEATURE_UNIT_LABEL, axisValue, fmtFeature, shortTicker } from '../lib/format'
import { INK, sequential } from '../lib/palette'
import { useApp, useFetch } from '../lib/store'

type Overlay = 'umatrix' | 'hits' | 'none'

export default function MapPage() {
  return <Requires run>{(runId) => <MapView runId={runId} />}</Requires>
}

function MapView({ runId }: { runId: string }) {
  const { data, error, loading } = useFetch(`map-${runId}`, () => api.map(runId))
  const [selected, setSelected] = useState<string | null>(null)
  const [overlay, setOverlay] = useState<Overlay>('umatrix')
  const [showBounds, setShowBounds] = useState(true)
  const [query, setQuery] = useState('')

  if (error) return <ErrorBox message={error} />
  if (loading || !data) return <Loading tall label="Loading map…" />

  const sel = data.stocks.find((s) => s.ticker === selected) ?? null
  const match = query.trim().toUpperCase()
  const found = match ? data.stocks.find((s) => s.symbol.toUpperCase() === match || s.symbol.toUpperCase().startsWith(match)) : undefined

  return (
    <div>
      <PageHeader eyebrow="Model · trained map" title="The behaviour map">
        Every cell is a neuron; each stock sits on the neuron whose weight vector (prototype) is closest to its features.
        Neighbouring neurons hold similar prototypes, so distance on the grid ≈ difference in behaviour. Thick lines mark the
        boundaries between the <span className="num">{Math.max(...data.neuron_clusters.flat()) + 1}</span> clusters.
      </PageHeader>

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <Segmented value={overlay} onChange={setOverlay}
          options={[{ value: 'umatrix', label: 'U-matrix' }, { value: 'hits', label: 'Hit count' }, { value: 'none', label: 'Plain' }]} />
        <Segmented value={showBounds ? 'on' : 'off'} onChange={(v) => setShowBounds(v === 'on')}
          options={[{ value: 'on', label: 'Cluster boundaries' }, { value: 'off', label: 'Hide' }]} />
        <input value={query} onChange={(e) => { setQuery(e.target.value); const s = data.stocks.find((x) => x.symbol.toUpperCase().startsWith(e.target.value.trim().toUpperCase())); if (e.target.value && s) setSelected(s.ticker) }}
          placeholder="Find ticker (e.g. INFY)"
          className="num w-48 rounded-md border border-line bg-panel-2 px-2.5 py-1.5 text-[12.5px] text-ink outline-none placeholder:text-ink-3 focus:border-accent" />
        {query && !found && <span className="text-[12px] text-ink-3">No match</span>}
        <div className="ml-auto flex gap-3 text-[12px] text-ink-3">
          <span>QE <span className="num text-ink-2">{data.qe.toFixed(3)}</span></span>
          <span>TE <span className="num text-ink-2">{(data.te * 100).toFixed(1)}%</span></span>
          <Badge>{data.input_mode === 'raw' ? '7 raw features' : 'autoencoder codes'}</Badge>
        </div>
      </div>

      <div className="grid gap-5 2xl:grid-cols-[1fr_360px] xl:grid-cols-[1fr_330px]">
        <Card pad={false}>
          <div className="p-4">
            <MainGrid data={data} overlay={overlay} showBounds={showBounds} selected={sel} onSelect={setSelected} />
            <div className="mt-4 flex flex-wrap items-end justify-between gap-4 border-t border-line pt-4">
              <SectorLegend compact />
              <OverlayLegend data={data} overlay={overlay} />
            </div>
          </div>
        </Card>
        <SidePanel runId={runId} stock={sel} onClose={() => setSelected(null)} />
      </div>

      <Card className="mt-5" title="Component planes"
        subtitle={<>One plane per feature: the colour of each neuron is that feature's value in its prototype, in real units{data.planes_decoded ? ' (decoded from the 3-D latent space by the autoencoder\'s decoder)' : ''}. Every plane uses the same blue ramp stretched to <em>its own</em> min→max, so compare patterns between planes, not absolute shades. Similar-looking planes = correlated features.</>}>
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {data.component_planes.map((p) => <Plane key={p.feature} plane={p} data={data} selected={sel} showBounds={showBounds} />)}
        </div>
      </Card>
    </div>
  )
}

function MainGrid({ data, overlay, showBounds, selected, onSelect }: {
  data: MapData; overlay: Overlay; showBounds: boolean; selected: MapStock | null; onSelect: (t: string) => void
}) {
  const { theme, palette } = useApp()
  const ink = INK[theme]
  const tip = useTooltip()
  const byCell = useMemo(() => {
    const m = new Map<string, MapStock[]>()
    for (const s of data.stocks) {
      const k = `${s.row},${s.col}`
      m.set(k, [...(m.get(k) ?? []), s])
    }
    return m
  }, [data])
  const umax = d3.max(data.umatrix.flat()) ?? 1
  const hmax = d3.max(data.hits.flat()) ?? 1
  const uS = sequential(theme, [0, umax])
  const hS = sequential(theme, [0, hmax])
  const fill = (r: number, c: number) =>
    overlay === 'umatrix' ? uS(data.umatrix[r][c]) : overlay === 'hits' ? hS(data.hits[r][c]) : ink.panel
  const textInk = (r: number, c: number) => (d3.lab(fill(r, c)).l > 60 ? '#121212' : '#f3f2ee')
  const W = 920
  const maxLines = data.cols <= 7 ? 6 : data.cols <= 9 ? 4 : 2
  return (
    <SomGrid rows={data.rows} cols={data.cols} width={W} fill={fill} axisLabels gap={3} radius={4}
      clusters={showBounds ? data.neuron_clusters : null}
      highlight={selected ? [[selected.row, selected.col]] : []}
      onLeave={tip.hide}
      onHover={(r, c, e) => {
        const list = byCell.get(`${r},${c}`) ?? []
        tip.show(e, (
          <div className="space-y-0.5">
            <div className="mb-1 flex justify-between gap-4 font-semibold"><span>Neuron ({r}, {c})</span><span className="num text-ink-3">cluster C{data.neuron_clusters[r][c]}</span></div>
            <TipRow label="U-matrix" value={data.umatrix[r][c].toFixed(3)} />
            <TipRow label="Stocks here" value={list.length} />
            {list.length > 0 && <div className="mt-1.5 space-y-0.5 border-t border-line pt-1.5">
              {list.map((s) => <TipRow key={s.ticker} color={palette.color(s.industry)} label={<span className="num">{s.symbol}</span>} value={<span className="font-sans text-ink-3">{s.industry.length > 22 ? s.industry.slice(0, 21) + '…' : s.industry}</span>} />)}
            </div>}
          </div>
        ))
      }}
      onClick={(r, c) => { const l = byCell.get(`${r},${c}`); if (l?.length) onSelect(l[0].ticker) }}
      renderCell={(r, c, b) => {
        const list = byCell.get(`${r},${c}`) ?? []
        const shown = list.slice(0, list.length > maxLines ? maxLines - 1 : maxLines)
        const lh = Math.min(15, (b.h - 10) / maxLines)
        const fs = Math.min(11.5, lh * 0.82)
        return (
          <g>
            <text x={b.x + b.w - 5} y={b.y + 11} textAnchor="end" fontSize={8.5} className="num" style={{ fill: textInk(r, c), opacity: 0.55 }}>C{data.neuron_clusters[r][c]}</text>
            {shown.map((s, i) => (
              <g key={s.ticker} onClick={(e) => { e.stopPropagation(); onSelect(s.ticker) }} style={{ cursor: 'pointer' }}>
                <rect x={b.x + 6} y={b.y + 7 + i * lh + lh / 2 - 3.5} width={7} height={7} rx={1.5} fill={palette.color(s.industry)}
                  stroke={textInk(r, c) === '#121212' ? 'rgba(0,0,0,.25)' : 'rgba(255,255,255,.2)'} strokeWidth={0.75} />
                <text x={b.x + 17} y={b.y + 7 + i * lh + lh / 2 + fs * 0.36} fontSize={fs} className="num"
                  style={{ fill: textInk(r, c), fontWeight: selected?.ticker === s.ticker ? 700 : 500 }}>
                  {s.symbol.length > 11 ? s.symbol.slice(0, 10) + '…' : s.symbol}
                </text>
              </g>
            ))}
            {list.length > shown.length && (
              <text x={b.x + 17} y={b.y + 7 + shown.length * lh + lh / 2 + fs * 0.36} fontSize={fs * 0.92} style={{ fill: textInk(r, c), opacity: 0.7 }}>
                +{list.length - shown.length} more
              </text>
            )}
          </g>
        )
      }}
      ariaLabel="SOM hit map with tickers" />
  )
}

function OverlayLegend({ data, overlay }: { data: MapData; overlay: Overlay }) {
  const { theme } = useApp()
  if (overlay === 'none') return <span className="text-[11.5px] text-ink-3">Cell shading off</span>
  const max = overlay === 'umatrix' ? d3.max(data.umatrix.flat()) ?? 1 : d3.max(data.hits.flat()) ?? 1
  return (
    <ColorRamp scale={sequential(theme, [0, max])} domain={[0, max]} width={180}
      label={overlay === 'umatrix' ? 'U-matrix: mean distance to neighbours (z-units) · dark = similar, bright = boundary' : 'Stocks mapped to neuron'}
      format={(v) => (overlay === 'umatrix' ? v.toFixed(2) : v.toFixed(0))} />
  )
}

function Plane({ plane, data, selected, showBounds }: {
  plane: MapData['component_planes'][number]; data: MapData; selected: MapStock | null; showBounds: boolean
}) {
  const { theme } = useApp()
  const tip = useTooltip()
  const vals = plane.values.flat()
  const dom: [number, number] = [d3.min(vals)!, d3.max(vals)!]
  const s = sequential(theme, dom)
  const f = plane.feature
  const fmt = (v: number) => {
    const a = axisValue(f, v)
    return f === 'Log_Traded_Value' ? `₹${a.toFixed(0)}Cr` : ['Annual_Return', 'Volatility', 'Max_Drawdown'].includes(f) ? `${a.toFixed(0)}%` : a.toFixed(2)
  }
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="text-[12.5px] font-medium text-ink">{FEATURE_LABEL[f]}</span>
        <span className="text-[11px] text-ink-3">{FEATURE_UNIT_LABEL[f]}</span>
      </div>
      <SomGrid rows={data.rows} cols={data.cols} width={240} gap={1.5} radius={2}
        fill={(r, c) => s(plane.values[r][c])}
        clusters={showBounds ? data.neuron_clusters : null}
        ring={selected ? [selected.row, selected.col] : null}
        onHover={(r, c, e) => tip.show(e, <div><div className="mb-1 font-semibold">{FEATURE_LABEL[f]} · ({r}, {c})</div><TipRow label="Prototype value" value={fmtFeature(f, plane.values[r][c])} /></div>)}
        onLeave={tip.hide} ariaLabel={`Component plane ${f}`} />
      <div className="mt-1.5"><ColorRamp scale={s} domain={dom} format={fmt} width={240} /></div>
    </div>
  )
}

function SidePanel({ runId, stock, onClose }: { runId: string; stock: MapStock | null; onClose: () => void }) {
  const { stocks } = useApp()
  const detail = useFetch(stock ? `stock-${runId}-${stock.ticker}` : null, () => api.stock(runId, stock!.ticker))
  const sectorZ = useMemo(() => {
    if (!stock || !stocks) return null
    const members = stocks.stocks.filter((s) => s.industry === stock.industry)
    return Object.fromEntries(stocks.features.map((f) => [f, d3.median(members, (m) => m.z[f] ?? NaN) ?? 0]))
  }, [stock, stocks])

  if (!stock) {
    return (
      <Card title="Stock detail">
        <p className="text-[13px] leading-relaxed text-ink-3">Click a ticker on the map (or search for one) to compare it with its sector median and its cluster's profile. Its neuron is then outlined on every component plane.</p>
      </Card>
    )
  }
  const d = detail.data
  const zdiff = d && sectorZ ? Object.fromEntries(Object.keys(sectorZ).map((f) => [f, (d.z[f] ?? 0) - sectorZ[f]])) : null
  return (
    <Card title={<span className="num text-[15px]">{shortTicker(stock.ticker)}</span>}
      subtitle={stock.company}
      right={<button onClick={onClose} className="text-[12px] text-ink-3 hover:text-ink">Close ✕</button>}>
      <div className="mb-4 flex flex-wrap gap-1.5">
        <Badge><SectorDot industry={stock.industry} />{stock.industry}</Badge>
        <Badge tone="accent">Cluster C{stock.cluster}</Badge>
        <Badge>Neuron ({stock.row}, {stock.col})</Badge>
      </div>
      {detail.error ? <ErrorBox message={detail.error} /> : !d ? <Loading /> : (
        <>
          <table className="w-full text-[12px]">
            <thead className="text-[11px] text-ink-3">
              <tr className="border-b border-line">
                <th className="py-1.5 text-left font-medium">Feature</th>
                <th className="py-1.5 text-right font-medium">Stock</th>
                <th className="py-1.5 text-right font-medium">Sector med.<div className="num font-normal">n={d.sector_size}</div></th>
                <th className="py-1.5 text-right font-medium">Cluster mean<div className="num font-normal">n={d.cluster_size}</div></th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(d.features).map((f) => (
                <tr key={f} className="border-b border-line/60">
                  <td className="py-1.5 text-ink-2">{FEATURE_LABEL[f]}</td>
                  <td className="num py-1.5 text-right font-medium text-ink">{fmtFeature(f, d.features[f])}</td>
                  <td className="num py-1.5 text-right text-ink-2">{fmtFeature(f, d.sector_median[f])}</td>
                  <td className="num py-1.5 text-right text-ink-2">{fmtFeature(f, d.cluster_mean[f])}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {zdiff && (
            <div className="mt-4">
              <div className="mb-1 text-[12px] font-medium text-ink-2">Distance from sector median, in standard deviations</div>
              <ZBars z={zdiff} features={Object.keys(zdiff)} reference={d.sector_median} refLabel="Sector median" />
              <p className="mt-1 text-[11px] text-ink-3">Red = above the sector, blue = below. Hover a bar for the real value.</p>
            </div>
          )}
        </>
      )}
    </Card>
  )
}

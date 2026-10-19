import * as d3 from 'd3'
import type { ReactNode } from 'react'
import { useApp } from '../lib/store'
import { INK, diverging, inkOn, sequential } from '../lib/palette'
import { FEATURE_SHORT, fmtFeature } from '../lib/format'
import type { DistSummary, FeatureDict } from '../api/types'
import { TipRow, useTooltip } from './ui'

/** Shared Recharts styling so every chart matches the theme. */
export function useChartStyle() {
  const { theme } = useApp()
  const ink = INK[theme]
  return {
    ink,
    tick: { fill: ink.text3, fontSize: 11, fontFamily: 'JetBrains Mono, monospace' },
    axisLabel: (value: string, angle = 0, position: 'insideBottom' | 'insideLeft' | 'insideRight' = angle ? 'insideLeft' : 'insideBottom') =>
      ({ value, angle, position, offset: angle ? 4 : -2, style: { fill: ink.text3, fontSize: 11, textAnchor: 'middle' as const } }),
    tooltip: {
      contentStyle: { background: ink.panel, border: `1px solid ${ink.border}`, borderRadius: 8, fontSize: 12, color: ink.text },
      labelStyle: { color: ink.text2, marginBottom: 4 },
      itemStyle: { padding: 0 },
      cursor: { stroke: ink.text3, strokeDasharray: '3 3' },
    },
    legend: { wrapperStyle: { fontSize: 12, color: ink.text2, paddingTop: 6 } },
  }
}

export function LegendRow({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-ink-2">
      {items.map((i) => (
        <span key={i.label} className="flex items-center gap-1.5">
          <svg width="16" height="8"><line x1="1" x2="15" y1="4" y2="4" stroke={i.color} strokeWidth="2" strokeDasharray={i.dashed ? '3 2' : undefined} strokeLinecap="round" /></svg>
          {i.label}
        </span>
      ))}
    </div>
  )
}

// ------------------------------------------------------------------ box + strip plot
export function BoxPlot({ groups, format, axisLabel, height = 230 }: {
  groups: { label: string; color: string; d: DistSummary }[]
  format: (v: number) => string
  axisLabel: string
  height?: number
}) {
  const { ink } = useChartStyle()
  const tip = useTooltip()
  const width = 420
  const m = { top: 12, right: 16, bottom: 30, left: 56 }
  const lo = d3.min(groups, (g) => g.d.min)!
  const hi = d3.max(groups, (g) => g.d.max)!
  const y = d3.scaleLinear().domain([lo, hi]).nice().range([height - m.bottom, m.top])
  const x = d3.scaleBand<string>().domain(groups.map((g) => g.label)).range([m.left, width - m.right]).padding(0.35)
  const bw = x.bandwidth()
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full">
      {y.ticks(5).map((t) => (
        <g key={t}>
          <line x1={m.left} x2={width - m.right} y1={y(t)} y2={y(t)} stroke={ink.grid} />
          <text x={m.left - 8} y={y(t) + 3.5} textAnchor="end" fontSize={10.5} className="num" style={{ fill: ink.text3 }}>{format(t)}</text>
        </g>
      ))}
      <text transform={`translate(13 ${(height - m.bottom + m.top) / 2}) rotate(-90)`} textAnchor="middle" fontSize={11} style={{ fill: ink.text3 }}>{axisLabel}</text>
      {groups.map((g, gi) => {
        const cx = x(g.label)! + bw / 2
        const rng = d3.randomLcg(gi + 1)
        return (
          <g key={g.label}>
            {/* strip of individual portfolios (a sample), jittered, very light */}
            {g.d.sample.map((v, i) => (
              <circle key={i} cx={cx + (rng() - 0.5) * bw * 0.9} cy={y(v)} r={1.6} fill={g.color} opacity={0.22} />
            ))}
            <line x1={cx} x2={cx} y1={y(g.d.whisker_low)} y2={y(g.d.q1)} stroke={ink.text2} strokeWidth={1.2} />
            <line x1={cx} x2={cx} y1={y(g.d.q3)} y2={y(g.d.whisker_high)} stroke={ink.text2} strokeWidth={1.2} />
            {[g.d.whisker_low, g.d.whisker_high].map((w, i) => (
              <line key={i} x1={cx - bw * 0.15} x2={cx + bw * 0.15} y1={y(w)} y2={y(w)} stroke={ink.text2} strokeWidth={1.2} />
            ))}
            <rect x={cx - bw * 0.32} width={bw * 0.64} y={y(g.d.q3)} height={Math.max(1, y(g.d.q1) - y(g.d.q3))}
              rx={4} fill={g.color} fillOpacity={0.28} stroke={g.color} strokeWidth={1.5}
              onMouseMove={(e) => tip.show(e, (
                <div className="space-y-0.5">
                  <div className="mb-1 font-semibold">{g.label}</div>
                  <TipRow label="Upper whisker" value={format(g.d.whisker_high)} />
                  <TipRow label="Q3" value={format(g.d.q3)} />
                  <TipRow label="Median" value={format(g.d.median)} />
                  <TipRow label="Q1" value={format(g.d.q1)} />
                  <TipRow label="Lower whisker" value={format(g.d.whisker_low)} />
                  <TipRow label="Mean" value={format(g.d.mean)} />
                </div>
              ))}
              onMouseLeave={tip.hide} />
            <line x1={cx - bw * 0.32} x2={cx + bw * 0.32} y1={y(g.d.median)} y2={y(g.d.median)} stroke={ink.text} strokeWidth={2.2} pointerEvents="none" />
            <text x={cx} y={height - m.bottom + 18} textAnchor="middle" fontSize={11.5} style={{ fill: ink.text2 }}>{g.label}</text>
          </g>
        )
      })}
    </svg>
  )
}

// ------------------------------------------------------------------ heatmap table
export function Heatmap({ rowLabels, colLabels, values, rowTitle, colTitle, cellTip, highlightDiagonal, rotateCols = true }: {
  rowLabels: string[]; colLabels: string[]; values: number[][]
  rowTitle: string; colTitle: string
  cellTip?: (r: number, c: number) => ReactNode
  highlightDiagonal?: boolean
  rotateCols?: boolean
}) {
  const { theme } = useApp()
  const ink = INK[theme]
  const tip = useTooltip()
  const max = d3.max(values.flat()) || 1
  const color = sequential(theme, [0, max])
  const cw = 34, ch = 26
  const left = 190
  const top = rotateCols ? 128 : 30
  const width = left + colLabels.length * cw + 4
  const height = top + rowLabels.length * ch + 4
  return (
    <div className="scroll-thin overflow-x-auto">
      <svg width={width} height={height} className="block">
        <text x={left - 8} y={top - 8} textAnchor="end" fontSize={10.5} style={{ fill: ink.text3 }}>{rowTitle} ↓ · {colTitle} →</text>
        {colLabels.map((l, c) => (
          <text key={l} fontSize={10.5} style={{ fill: ink.text2 }}
            transform={rotateCols ? `translate(${left + c * cw + cw / 2 + 3} ${top - 6}) rotate(-55)` : `translate(${left + c * cw + cw / 2} ${top - 8})`}
            textAnchor={rotateCols ? 'start' : 'middle'}>
            {l.length > 22 ? `${l.slice(0, 21)}…` : l}
          </text>
        ))}
        {rowLabels.map((l, r) => (
          <g key={l}>
            <text x={left - 8} y={top + r * ch + ch / 2 + 4} textAnchor="end" fontSize={11} style={{ fill: ink.text2 }}>
              {l.length > 27 ? `${l.slice(0, 26)}…` : l}
            </text>
            {colLabels.map((_, c) => {
              const v = values[r][c]
              const f = v === 0 ? 'transparent' : color(v)
              return (
                <g key={c} onMouseMove={(e) => cellTip && tip.show(e, cellTip(r, c))} onMouseLeave={tip.hide}>
                  <rect x={left + c * cw + 1} y={top + r * ch + 1} width={cw - 2} height={ch - 2} rx={3}
                    fill={f} stroke={ink.grid} strokeWidth={v === 0 ? 1 : 0} />
                  {highlightDiagonal && r === c && (
                    <rect x={left + c * cw + 1} y={top + r * ch + 1} width={cw - 2} height={ch - 2} rx={3} fill="none" stroke={ink.accent} strokeWidth={1.5} />
                  )}
                  {v > 0 && (
                    <text x={left + c * cw + cw / 2} y={top + r * ch + ch / 2 + 4} textAnchor="middle" fontSize={11}
                      className="num" style={{ fill: inkOn(f) }}>{v}</text>
                  )}
                </g>
              )
            })}
          </g>
        ))}
      </svg>
    </div>
  )
}

// ------------------------------------------------------------------ z-difference bars
/** Horizontal diverging bars: how many std-devs a stock sits above/below a reference. */
export function ZBars({ z, features, reference, refLabel, compact }: {
  z: FeatureDict; features: string[]; reference?: FeatureDict; refLabel: string; compact?: boolean
}) {
  const { theme } = useApp()
  const ink = INK[theme]
  const tip = useTooltip()
  const maxAbs = 3
  const color = diverging(theme, maxAbs)
  const width = compact ? 260 : 360
  const left = compact ? 62 : 86
  const rowH = compact ? 15 : 20
  const x = d3.scaleLinear().domain([-maxAbs, maxAbs]).range([left, width - 8]).clamp(true)
  const height = features.length * rowH + 16
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full">
      <line x1={x(0)} x2={x(0)} y1={0} y2={height - 14} stroke={ink.text3} strokeWidth={1} />
      {features.map((f, i) => {
        const v = z[f] ?? 0
        const yy = i * rowH
        return (
          <g key={f} onMouseMove={(e) => tip.show(e, (
            <div className="space-y-0.5">
              <div className="mb-1 font-semibold">{FEATURE_SHORT[f]}</div>
              <TipRow label="Difference" value={`${v >= 0 ? '+' : ''}${v.toFixed(2)} σ`} />
              {reference && <TipRow label={refLabel} value={fmtFeature(f, reference[f])} />}
            </div>
          ))} onMouseLeave={tip.hide}>
            <rect x={0} y={yy} width={width} height={rowH} fill="transparent" />
            <text x={left - 6} y={yy + rowH / 2 + 3.5} textAnchor="end" fontSize={compact ? 9.5 : 11} style={{ fill: ink.text2 }}>{FEATURE_SHORT[f]}</text>
            <rect x={Math.min(x(0), x(v))} y={yy + 3} width={Math.max(1, Math.abs(x(v) - x(0)))} height={rowH - 6} rx={2}
              fill={color(v)} />
          </g>
        )
      })}
      {[-2, 0, 2].map((t) => (
        <text key={t} x={x(t)} y={height - 3} textAnchor="middle" fontSize={9} className="num" style={{ fill: ink.text3 }}>{t > 0 ? `+${t}σ` : `${t}σ`}</text>
      ))}
    </svg>
  )
}

import * as d3 from 'd3'
import { useMemo, type ReactNode, type MouseEvent } from 'react'
import type { Grid } from '../api/types'
import { useApp } from '../lib/store'
import { INK } from '../lib/palette'

/*
  A rectangular SOM drawn as an SVG grid. d3 band scales place the neurons; the caller
  decides what colours each cell (U-matrix, a component plane, hits ...) and what is
  drawn inside it. Cluster boundaries are drawn as thick lines between neighbouring
  neurons that belong to different K-means clusters.
*/
export interface SomGridProps {
  rows: number
  cols: number
  width: number                      // logical width; the SVG scales to its container
  fill?: (r: number, c: number) => string
  clusters?: Grid | null
  highlight?: [number, number][]
  highlightColor?: string
  ring?: [number, number] | null     // e.g. BMU marker
  renderCell?: (r: number, c: number, box: { x: number; y: number; w: number; h: number }) => ReactNode
  onHover?: (r: number, c: number, e: MouseEvent) => void
  onLeave?: () => void
  onClick?: (r: number, c: number) => void
  gap?: number
  radius?: number
  axisLabels?: boolean
  ariaLabel?: string
}

export function SomGrid({
  rows, cols, width, fill, clusters, highlight = [], highlightColor, ring, renderCell,
  onHover, onLeave, onClick, gap = 2, radius = 3, axisLabels = false, ariaLabel,
}: SomGridProps) {
  const { theme } = useApp()
  const ink = INK[theme]
  const pad = axisLabels ? 16 : 0
  const cell = (width - pad) / cols
  const height = cell * rows + pad
  const x = useMemo(() => d3.scaleBand<number>().domain(d3.range(cols)).range([pad, width]).paddingInner(0), [cols, width, pad])
  const y = useMemo(() => d3.scaleBand<number>().domain(d3.range(rows)).range([pad, height]).paddingInner(0), [rows, height, pad])

  const boundaries = useMemo(() => {
    if (!clusters) return []
    const segs: [number, number, number, number][] = []
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (c + 1 < cols && clusters[r][c] !== clusters[r][c + 1]) {
          const xx = x(c + 1)!
          segs.push([xx, y(r)!, xx, y(r)! + cell])
        }
        if (r + 1 < rows && clusters[r][c] !== clusters[r + 1][c]) {
          const yy = y(r + 1)!
          segs.push([x(c)!, yy, x(c)! + cell, yy])
        }
      }
    }
    return segs
  }, [clusters, rows, cols, x, y, cell])

  const hl = new Set(highlight.map(([r, c]) => `${r},${c}`))

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="block h-auto w-full select-none" role="img" aria-label={ariaLabel}>
      {axisLabels && d3.range(cols).map((c) => (
        <text key={`c${c}`} x={x(c)! + cell / 2} y={10} textAnchor="middle" fontSize={9} className="num" style={{ fill: ink.text3 }}>{c}</text>
      ))}
      {axisLabels && d3.range(rows).map((r) => (
        <text key={`r${r}`} x={6} y={y(r)! + cell / 2 + 3} textAnchor="middle" fontSize={9} className="num" style={{ fill: ink.text3 }}>{r}</text>
      ))}
      {d3.range(rows).map((r) => d3.range(cols).map((c) => {
        const bx = x(c)!, by = y(r)!
        const box = { x: bx + gap / 2, y: by + gap / 2, w: cell - gap, h: cell - gap }
        return (
          <g key={`${r}-${c}`}
            onMouseMove={onHover ? (e) => onHover(r, c, e) : undefined}
            onMouseLeave={onLeave}
            onClick={onClick ? () => onClick(r, c) : undefined}
            style={{ cursor: onClick ? 'pointer' : undefined }}>
            <rect {...{ x: box.x, y: box.y, width: box.w, height: box.h }} rx={radius}
              style={{ fill: fill ? fill(r, c) : ink.panel, transition: 'fill 220ms linear' }} />
            {renderCell?.(r, c, box)}
            {hl.has(`${r},${c}`) && (
              <rect x={box.x + 1} y={box.y + 1} width={box.w - 2} height={box.h - 2} rx={radius} fill="none"
                style={{ stroke: highlightColor ?? ink.accent }} strokeWidth={2.5} />
            )}
          </g>
        )
      }))}
      {boundaries.map(([x1, y1, x2, y2], i) => (
        <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} style={{ stroke: ink.text }} strokeWidth={2.5} strokeLinecap="round" pointerEvents="none" />
      ))}
      {ring && (
        <rect x={x(ring[1])! + 1} y={y(ring[0])! + 1} width={cell - 2} height={cell - 2} rx={radius + 1} fill="none"
          style={{ stroke: ink.accent }} strokeWidth={3} pointerEvents="none" />
      )}
    </svg>
  )
}

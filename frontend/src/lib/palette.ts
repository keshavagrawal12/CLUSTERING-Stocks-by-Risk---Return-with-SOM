import * as d3 from 'd3'

export type Theme = 'dark' | 'light'

/*
  Validated 8-slot categorical palette (adjacent pairs pass colour-blind ΔE checks).
  Hues are assigned in this FIXED order and never cycled: the 8 largest sectors get a
  slot (by stock count, ties broken alphabetically), every smaller sector shares the
  neutral "other" grey. Sector identity never relies on colour alone - tickers, legends
  and tooltips always name the sector too.
*/
export const SERIES: Record<Theme, string[]> = {
  light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'],
  dark: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
}
export const OTHER: Record<Theme, string> = { light: '#a9a7a0', dark: '#6b6a66' }

const SEQ: Record<Theme, [string, string]> = { light: ['#eef4fc', '#0d366b'], dark: ['#17202c', '#9ec5f4'] }
const DIV: Record<Theme, [string, string, string]> = {
  light: ['#2a78d6', '#f0efec', '#e34948'],
  dark: ['#3987e5', '#383835', '#e66767'],
}

export const INK: Record<Theme, { text: string; text2: string; text3: string; grid: string; accent: string; panel: string; border: string }> = {
  dark: { text: '#ecebe6', text2: '#b4b2aa', text3: '#7d7b74', grid: '#22242a', accent: '#d4b26a', panel: '#15161a', border: '#26282e' },
  light: { text: '#121212', text2: '#52514e', text3: '#85837c', grid: '#ecebe6', accent: '#8a6516', panel: '#fcfcfb', border: '#e2e0d9' },
}

export interface SectorPalette {
  color: (industry: string) => string
  coloured: string[]          // sectors that got their own hue, in slot order
  other: string[]             // sectors folded into grey
}

export function buildSectorPalette(counts: Record<string, number>, theme: Theme): SectorPalette {
  const ordered = Object.keys(counts).sort((a, b) => counts[b] - counts[a] || a.localeCompare(b))
  const coloured = ordered.slice(0, 8)
  const other = ordered.slice(8)
  const map = new Map(coloured.map((s, i) => [s, SERIES[theme][i]]))
  return { color: (s) => map.get(s) ?? OTHER[theme], coloured, other }
}

/** Sequential single-hue ramp: low values recede toward the surface. */
export function sequential(theme: Theme, domain: [number, number]) {
  return d3.scaleSequential(d3.interpolateLab(SEQ[theme][0], SEQ[theme][1])).domain(domain)
}

/** Diverging ramp with a neutral grey midpoint at 0. */
export function diverging(theme: Theme, maxAbs: number) {
  const [neg, mid, pos] = DIV[theme]
  return d3.scaleDiverging(d3.piecewise(d3.interpolateLab, [neg, mid, pos])).domain([-maxAbs, 0, maxAbs])
}

/** Readable ink on top of a filled cell. */
export function inkOn(fill: string) {
  const c = d3.lab(fill)
  return c.l > 60 ? '#121212' : '#f3f2ee'
}

/** Rule colours for the two RQ3 portfolio rules + the index. */
export function ruleColors(theme: Theme) {
  return { cluster: SERIES[theme][0], sector: SERIES[theme][1], nifty: OTHER[theme] }
}

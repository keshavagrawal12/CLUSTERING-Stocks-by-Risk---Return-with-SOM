// Formatting helpers: every number is shown in its real unit (%, ×, ₹).

export const FEATURE_LABEL: Record<string, string> = {
  Annual_Return: 'Annual return',
  Volatility: 'Volatility',
  Beta: 'Beta',
  Correlation: 'Correlation',
  Max_Drawdown: 'Max drawdown',
  Sharpe_Ratio: 'Sharpe ratio',
  Log_Traded_Value: 'Traded value',
}

export const FEATURE_SHORT: Record<string, string> = {
  Annual_Return: 'Return',
  Volatility: 'Vol',
  Beta: 'Beta',
  Correlation: 'Corr',
  Max_Drawdown: 'MaxDD',
  Sharpe_Ratio: 'Sharpe',
  Log_Traded_Value: 'Traded val',
}

export const FEATURE_UNIT_LABEL: Record<string, string> = {
  Annual_Return: '% p.a.',
  Volatility: '% p.a.',
  Beta: '× NIFTY',
  Correlation: 'ρ',
  Max_Drawdown: '%',
  Sharpe_Ratio: 'ratio',
  Log_Traded_Value: '₹ Cr / day',
}

export function pct(v: number | null | undefined, digits = 1) {
  if (v == null || !Number.isFinite(v)) return '—'
  return `${(v * 100).toFixed(digits)}%`
}

export function num(v: number | null | undefined, digits = 2) {
  if (v == null || !Number.isFinite(v)) return '—'
  return v.toFixed(digits)
}

/** log(mean daily ₹ traded) -> "₹ 1,234 Cr" */
export function crore(logValue: number | null | undefined) {
  if (logValue == null || !Number.isFinite(logValue)) return '—'
  const cr = Math.exp(logValue) / 1e7
  return `₹${cr >= 100 ? Math.round(cr).toLocaleString('en-IN') : cr.toFixed(1)} Cr`
}

export function fmtFeature(feature: string, v: number | null | undefined) {
  switch (feature) {
    case 'Annual_Return':
    case 'Volatility':
    case 'Max_Drawdown':
      return pct(v)
    case 'Beta':
      return v == null ? '—' : `${num(v)}×`
    case 'Log_Traded_Value':
      return crore(v)
    default:
      return num(v)
  }
}

/** Value used on chart axes for a feature (percent features scaled ×100, traded value in ₹ Cr). */
export function axisValue(feature: string, v: number) {
  if (['Annual_Return', 'Volatility', 'Max_Drawdown'].includes(feature)) return v * 100
  if (feature === 'Log_Traded_Value') return Math.exp(v) / 1e7
  return v
}

export function pValue(p: number) {
  if (p < 1e-4) return p.toExponential(1)
  return p.toFixed(4)
}

export const shortTicker = (t: string) => t.replace(/\.NS$/, '')

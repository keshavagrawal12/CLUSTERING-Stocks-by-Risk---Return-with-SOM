import { createContext, useCallback, useContext, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { useApp } from '../lib/store'
import { navigate } from '../lib/router'

export function PageHeader({ eyebrow, title, children }: { eyebrow?: string; title: string; children?: ReactNode }) {
  return (
    <header className="mb-8 max-w-3xl">
      {eyebrow && <div className="mb-2 text-[11px] font-medium uppercase tracking-[0.14em] text-accent">{eyebrow}</div>}
      <h1 className="text-[26px] font-semibold leading-tight tracking-tight text-ink">{title}</h1>
      {children && <div className="mt-3 text-[14.5px] leading-relaxed text-ink-2">{children}</div>}
    </header>
  )
}

export function Card({ title, subtitle, right, children, className = '', pad = true }: {
  title?: ReactNode; subtitle?: ReactNode; right?: ReactNode; children: ReactNode; className?: string; pad?: boolean
}) {
  return (
    <section className={`rounded-xl border border-line bg-panel ${className}`}>
      {(title || right) && (
        <div className="flex items-start justify-between gap-4 border-b border-line px-5 py-3.5">
          <div>
            {title && <h2 className="text-[13.5px] font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-[12.5px] leading-snug text-ink-3">{subtitle}</p>}
          </div>
          {right}
        </div>
      )}
      <div className={pad ? 'p-5' : ''}>{children}</div>
    </section>
  )
}

export function Stat({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: 'accent' | 'muted' }) {
  return (
    <div>
      <div className="text-[11.5px] font-medium uppercase tracking-[0.08em] text-ink-3">{label}</div>
      <div className={`num mt-1 text-[24px] font-medium leading-none ${tone === 'accent' ? 'text-accent' : 'text-ink'}`}>{value}</div>
      {sub && <div className="mt-1.5 text-[12px] leading-snug text-ink-3">{sub}</div>}
    </div>
  )
}

export function Badge({ children, tone = 'default' }: { children: ReactNode; tone?: 'default' | 'accent' | 'critical' | 'good' }) {
  const cls = {
    default: 'border-line-strong text-ink-2',
    accent: 'border-accent/40 text-accent bg-accent-soft',
    critical: 'border-critical/40 text-critical',
    good: 'border-good/40 text-good',
  }[tone]
  return <span className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-medium ${cls}`}>{children}</span>
}

export function Button({ children, onClick, variant = 'primary', disabled, type = 'button', small }: {
  children: ReactNode; onClick?: () => void; variant?: 'primary' | 'ghost'; disabled?: boolean; type?: 'button' | 'submit'; small?: boolean
}) {
  const base = `inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${small ? 'px-3 py-1.5 text-[12.5px]' : 'px-4 py-2 text-[13.5px]'}`
  const v = variant === 'primary'
    ? 'bg-accent text-accent-ink hover:brightness-110'
    : 'border border-line-strong text-ink-2 hover:text-ink hover:border-ink-3'
  return <button type={type} className={`${base} ${v}`} onClick={onClick} disabled={disabled}>{children}</button>
}

export function Segmented<T extends string>({ value, options, onChange, disabled }: {
  value: T; options: { value: T; label: string; disabled?: boolean; title?: string }[]; onChange: (v: T) => void; disabled?: boolean
}) {
  return (
    <div className="inline-flex rounded-lg border border-line bg-panel-2 p-0.5">
      {options.map((o) => (
        <button key={o.value} type="button" title={o.title}
          disabled={disabled || o.disabled}
          onClick={() => onChange(o.value)}
          className={`rounded-md px-3 py-1 text-[12.5px] font-medium transition-colors disabled:opacity-40 ${value === o.value ? 'bg-panel text-ink shadow-sm ring-1 ring-line-strong' : 'text-ink-3 hover:text-ink-2'}`}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Spinner() {
  return <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-ink-3 border-t-transparent" />
}

export function Loading({ label = 'Loading…', tall }: { label?: string; tall?: boolean }) {
  return (
    <div className={`flex items-center justify-center gap-3 text-[13px] text-ink-3 ${tall ? 'h-72' : 'h-32'}`}>
      <Spinner /> {label}
    </div>
  )
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-critical/40 bg-critical/5 px-4 py-3 text-[13px] text-critical">
      <span className="font-semibold">Error · </span>{message}
    </div>
  )
}

export function EmptyState({ title, children, action }: { title: string; children: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-line-strong px-6 py-16 text-center">
      <svg width="44" height="44" viewBox="0 0 44 44" className="mb-5 opacity-60" aria-hidden>
        {[0, 1, 2].map((r) => [0, 1, 2].map((c) => (
          <rect key={`${r}${c}`} x={4 + c * 13} y={4 + r * 13} width="10" height="10" rx="2" fill="none" stroke="var(--text-3)" strokeWidth="1.2" />
        )))}
      </svg>
      <h3 className="text-[15px] font-semibold text-ink">{title}</h3>
      <div className="mt-2 max-w-md text-[13.5px] leading-relaxed text-ink-2">{children}</div>
      {action && <div className="mt-5">{action}</div>}
    </div>
  )
}

/** Wraps any page that needs data and/or a trained run, showing the right empty state. */
export function Requires({ run, children }: { run?: boolean; children: (runId: string) => ReactNode }) {
  const { status, statusError, runId } = useApp()
  if (statusError) return <EmptyState title="Backend not reachable">{statusError} Start it with <code className="num">./dev.sh</code> (macOS) or <code className="num">dev.ps1</code> (Windows).</EmptyState>
  if (!status) return <Loading tall />
  if (!status.loaded) {
    return (
      <EmptyState title="No data loaded"
        action={<Button onClick={() => navigate('data')}>Go to Data → upload CSVs</Button>}>
        Upload <span className="num">nifty100_features.csv</span>, <span className="num">nifty100_prices.csv</span> and <span className="num">nifty100_volumes.csv</span> on the Data page,
        or place them in <span className="num">backend/data/raw/</span>, or run <span className="num">python -m scripts.fetch_data</span>.
        {status.load_error && <div className="mt-3 text-critical">Last load error: {status.load_error}</div>}
      </EmptyState>
    )
  }
  if (run && !runId) {
    return (
      <EmptyState title="No trained map yet" action={<Button onClick={() => navigate('train')}>Train a SOM</Button>}>
        This view is computed from a trained Self-Organizing Map. Train one on the Train page (defaults take about a second).
      </EmptyState>
    )
  }
  return <>{children(runId ?? '')}</>
}

// ---------------------------------------------------------------- floating tooltip
interface TipState { x: number; y: number; content: ReactNode }
const TipCtx = createContext<{ show: (e: { clientX: number; clientY: number }, c: ReactNode) => void; hide: () => void } | null>(null)

export function TooltipProvider({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<TipState | null>(null)
  const show = useCallback((e: { clientX: number; clientY: number }, content: ReactNode) => setTip({ x: e.clientX, y: e.clientY, content }), [])
  const hide = useCallback(() => setTip(null), [])
  return (
    <TipCtx.Provider value={{ show, hide }}>
      {children}
      {tip && createPortal(
        <div className="pointer-events-none fixed z-50 max-w-xs rounded-lg border border-line-strong bg-panel px-3 py-2 text-[12.5px] leading-snug text-ink shadow-xl"
          style={{
            left: Math.min(tip.x + 14, window.innerWidth - 300),
            top: tip.y + 14 + 160 > window.innerHeight ? tip.y - 14 - 120 : tip.y + 14,
          }}>
          {tip.content}
        </div>, document.body)}
    </TipCtx.Provider>
  )
}

export function useTooltip() {
  const v = useContext(TipCtx)
  if (!v) throw new Error('useTooltip outside provider')
  return v
}

export function TipRow({ label, value, color }: { label: ReactNode; value: ReactNode; color?: string }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="flex items-center gap-1.5 text-ink-2">
        {color && <span className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />}
        {label}
      </span>
      <span className="num text-ink">{value}</span>
    </div>
  )
}

export function SectorDot({ industry, size = 8 }: { industry: string; size?: number }) {
  const { palette } = useApp()
  return <span className="inline-block shrink-0 rounded-full" style={{ width: size, height: size, background: palette.color(industry) }} />
}

export function SectorLegend({ compact }: { compact?: boolean }) {
  const { palette } = useApp()
  return (
    <div className={`flex flex-wrap gap-x-4 gap-y-1.5 ${compact ? 'text-[11.5px]' : 'text-[12px]'} text-ink-2`}>
      {palette.coloured.map((s) => (
        <span key={s} className="flex items-center gap-1.5"><SectorDot industry={s} />{s}</span>
      ))}
      {palette.other.length > 0 && (
        <span className="flex items-center gap-1.5" title={palette.other.join(', ')}>
          <SectorDot industry="__other__" />Other sectors ({palette.other.length})
        </span>
      )}
    </div>
  )
}

/** Continuous colour legend for a d3 scale. */
export function ColorRamp({ scale, domain, format, label, width = 160 }: {
  scale: (v: number) => string; domain: [number, number]; format: (v: number) => string; label?: string; width?: number
}) {
  const stops = Array.from({ length: 11 }, (_, i) => domain[0] + (i / 10) * (domain[1] - domain[0]))
  const id = `g${Math.round(Math.random() * 1e9)}`
  return (
    <div className="text-[11px] text-ink-3">
      {label && <div className="mb-1">{label}</div>}
      <svg width={width} height={10}>
        <defs>
          <linearGradient id={id}>
            {stops.map((s, i) => <stop key={i} offset={`${i * 10}%`} stopColor={scale(s)} />)}
          </linearGradient>
        </defs>
        <rect width={width} height={10} rx={2} fill={`url(#${id})`} />
      </svg>
      <div className="num mt-0.5 flex justify-between" style={{ width }}>
        <span>{format(domain[0])}</span><span>{format(domain[1])}</span>
      </div>
    </div>
  )
}

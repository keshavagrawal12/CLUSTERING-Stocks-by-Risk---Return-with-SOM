import * as d3 from 'd3'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, streamTraining } from '../api/client'
import type { AEProgress, Checkpoint, TrainParams } from '../api/types'
import { useChartStyle } from '../components/charts'
import { SomGrid } from '../components/SomGrid'
import { Button, Card, ColorRamp, ErrorBox, PageHeader, Requires, Segmented, Spinner } from '../components/ui'
import { inkOn, sequential, SERIES } from '../lib/palette'
import { navigate } from '../lib/router'
import { useApp } from '../lib/store'

const FRAME_MS = 110   // playback speed: training itself takes well under a second

function Field({ label, hint, children }: { label: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="block">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-[12.5px] font-medium text-ink-2">{label}</span>
        {hint && <span className="text-[11px] text-ink-3">{hint}</span>}
      </div>
      {children}
    </label>
  )
}

function NumInput({ value, onChange, step = 1, min, max }: { value: number; onChange: (v: number) => void; step?: number; min?: number; max?: number }) {
  return (
    <input type="number" value={value} step={step} min={min} max={max}
      onChange={(e) => onChange(Number(e.target.value))}
      className="num w-full rounded-md border border-line bg-panel-2 px-2.5 py-1.5 text-[13px] text-ink outline-none focus:border-accent" />
  )
}

function MiniLine({ data, dataKey, label, color, format, phaseAt, domain }: {
  data: { iteration: number }[]; dataKey: string; label: string; color: string; format: (v: number) => string; phaseAt: number; domain?: [number, number]
}) {
  const { tick, tooltip, axisLabel, ink } = useChartStyle()
  return (
    <div>
      <div className="mb-1 text-[12px] font-medium text-ink-2">{label}</div>
      <ResponsiveContainer width="100%" height={140}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 14, left: -6 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="iteration" type="number" domain={[0, 'dataMax']} tick={tick} tickFormatter={(v) => `${v / 1000}k`} label={axisLabel('iteration')} />
          <YAxis tick={tick} tickFormatter={format} width={46} domain={domain ?? ['auto', 'auto']} />
          <Tooltip {...tooltip} labelFormatter={(v) => `iteration ${Number(v).toLocaleString()}`} formatter={(v) => [format(Number(v)), label]} />
          <ReferenceLine x={phaseAt} stroke={ink.text3} strokeDasharray="3 3" />
          <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function TrainPage() {
  return <Requires>{() => <Trainer />}</Requires>
}

function Trainer() {
  const { status, runId, setRunId, theme } = useApp()
  const { tick, tooltip, axisLabel, legend } = useChartStyle()
  const [p, setP] = useState<TrainParams>(() => ({ ...status!.defaults, autoencoder: null }))
  const [frames, setFrames] = useState<Checkpoint[]>([])
  const [shown, setShown] = useState(0)
  const [ae, setAe] = useState<AEProgress | null>(null)
  const [phase, setPhase] = useState<'idle' | 'running' | 'done' | 'error'>('idle')
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const closeRef = useRef<() => void>(() => {})
  const trainedHere = useRef<string | null>(null)   // run streamed live on this page: don't reload its history

  // Show the currently selected run's history when arriving on the page.
  useEffect(() => {
    if (!runId || phase === 'running' || runId === trainedHere.current) return
    let alive = true
    fetch(`/api/som/${runId}/history`).then((r) => r.json()).then((h) => {
      if (!alive || !h.progress) return
      setFrames(h.progress.filter((e: { type: string }) => e.type === 'checkpoint'))
      setAe(h.progress.find((e: { type: string }) => e.type === 'autoencoder') ?? null)
      setP((cur) => ({ ...cur, ...h.params }))
      setShown(Number.MAX_SAFE_INTEGER)
      setPhase('done')
    }).catch(() => {})
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId])

  // Playback clock: step the displayed frame towards the newest received frame.
  useEffect(() => {
    if (shown >= frames.length - 1) return
    const t = setTimeout(() => setShown((s) => Math.min(s + 1, frames.length - 1)), FRAME_MS)
    return () => clearTimeout(t)
  }, [shown, frames.length])

  useEffect(() => () => closeRef.current(), [])

  const start = async () => {
    setError(null); setFrames([]); setShown(0); setAe(null); setMessage(null); setPhase('running')
    try {
      const { run_id } = await api.train(p)
      closeRef.current = streamTraining(run_id, (e) => {
        if (e.type === 'checkpoint') setFrames((f) => [...f, e])
        else if (e.type === 'autoencoder') { setAe(e); setMessage('Autoencoder trained, training SOM on the 3-D codes…') }
        else if (e.type === 'status') setMessage(e.message)
        else if (e.type === 'done') { trainedHere.current = e.run_id; setPhase('done'); setMessage(null); setRunId(e.run_id) }
        else if (e.type === 'error') { setPhase('error'); setError(e.message) }
      }, (msg) => { setPhase('error'); setError(msg) })
    } catch (e) { setPhase('error'); setError((e as Error).message) }
  }

  const n = status!.n_stocks
  const suggested = status!.suggested_neurons ?? Math.round(5 * Math.sqrt(n))
  const side = Math.round(Math.sqrt(suggested))
  const cur = frames[Math.min(shown, frames.length - 1)]
  const visible = frames.slice(0, Math.min(shown, frames.length - 1) + 1)
  const umax = useMemo(() => d3.max(frames.flatMap((f) => f.umatrix.flat())) ?? 1, [frames])
  const hmax = cur ? d3.max(cur.hits.flat()) ?? 1 : 1
  const uScale = sequential(theme, [0, umax])
  const hScale = sequential(theme, [0, Math.max(1, hmax)])
  const set = <K extends keyof TrainParams>(k: K, v: TrainParams[K]) => setP((s) => ({ ...s, [k]: v }))
  const playing = shown < frames.length - 1
  const s = SERIES[theme]

  return (
    <div>
      <PageHeader eyebrow="Model · Unit 3, Practical 8" title="Train the Self-Organizing Map">
        Each iteration picks one stock, finds its best-matching neuron (competition), and pulls that neuron and its grid
        neighbours towards the stock (cooperation + adaptation). Watch the U-matrix organise as the learning rate η and radius σ shrink.
      </PageHeader>

      <div className="grid gap-5 xl:grid-cols-[320px_1fr]">
        <Card title="Hyperparameters" subtitle="Defaults come from backend/app/config.py">
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Rows"><NumInput value={p.rows} min={2} max={20} onChange={(v) => set('rows', v)} /></Field>
              <Field label="Cols"><NumInput value={p.cols} min={2} max={20} onChange={(v) => set('cols', v)} /></Field>
            </div>
            <div className="rounded-md bg-accent-soft px-3 py-2 text-[12px] leading-snug text-ink-2">
              Rule of thumb: <span className="num text-ink">5·√N = 5·√{n} ≈ {suggested}</span> neurons → about <span className="num text-ink">{side}×{side}</span>.
              Current: <span className="num text-ink">{p.rows * p.cols}</span>.
            </div>
            <div className="grid grid-cols-2 gap-3">
              <Field label="σ₀ (radius)"><NumInput value={p.sigma0} step={0.1} onChange={(v) => set('sigma0', v)} /></Field>
              <Field label="η₀ (learning rate)"><NumInput value={p.eta0} step={0.05} onChange={(v) => set('eta0', v)} /></Field>
              <Field label="Ordering iters"><NumInput value={p.ordering_iters} step={500} onChange={(v) => set('ordering_iters', v)} /></Field>
              <Field label="Convergence iters"><NumInput value={p.convergence_iters} step={500} onChange={(v) => set('convergence_iters', v)} /></Field>
            </div>
            <Field label="Initialisation">
              <Segmented value={p.init} onChange={(v) => set('init', v)}
                options={[{ value: 'pca', label: 'PCA' }, { value: 'random', label: 'Random' }]} />
            </Field>
            <Field label="Input" hint={status!.autoencoder_available ? undefined : 'TensorFlow missing'}>
              <Segmented value={p.input_mode} onChange={(v) => set('input_mode', v)}
                options={[
                  { value: 'raw', label: '7 raw features' },
                  { value: 'autoencoder', label: 'AE 3-D codes', disabled: !status!.autoencoder_available, title: status!.autoencoder_reason ?? undefined },
                ]} />
            </Field>
            <Field label="Random seed"><NumInput value={p.seed} onChange={(v) => set('seed', v)} /></Field>
            <div className="flex gap-2 pt-1">
              <Button onClick={start} disabled={phase === 'running'}>{phase === 'running' && <Spinner />}Train</Button>
              <Button variant="ghost" onClick={() => setP({ ...status!.defaults, autoencoder: null })} disabled={phase === 'running'}>Reset</Button>
            </div>
            {error && <ErrorBox message={error} />}
          </div>
        </Card>

        <div className="space-y-5">
          <Card title="Map organising"
            subtitle={cur ? `${cur.phase === 'ordering' ? 'Ordering phase' : 'Convergence phase'} · showing snapshot ${Math.min(shown, frames.length - 1) + 1} of ${frames.length}` : 'Press Train to start'}
            right={phase === 'done' && !playing && frames.length > 0 ? (
              <div className="flex gap-2">
                <Button small variant="ghost" onClick={() => setShown(0)}>Replay</Button>
                <Button small onClick={() => navigate('map')}>Open map →</Button>
              </div>
            ) : undefined}>
            {message && <div className="mb-3 flex items-center gap-2 text-[12.5px] text-ink-2"><Spinner />{message}</div>}
            {!cur ? (
              <div className="flex h-64 items-center justify-center text-[13px] text-ink-3">
                {phase === 'running' ? <span className="flex items-center gap-2"><Spinner />Waiting for the first snapshot…</span> : 'No training yet.'}
              </div>
            ) : (
              <>
                <div className="mb-4 grid grid-cols-2 gap-4 sm:grid-cols-5">
                  {[
                    ['Iteration', `${cur.iteration.toLocaleString()} / ${cur.total.toLocaleString()}`],
                    ['η(t)', cur.eta.toFixed(4)], ['σ(t)', cur.sigma.toFixed(3)],
                    ['QE', cur.qe.toFixed(3)], ['TE', `${(cur.te * 100).toFixed(1)}%`],
                  ].map(([k, v]) => (
                    <div key={k}><div className="text-[11px] uppercase tracking-[0.08em] text-ink-3">{k}</div><div className="num mt-0.5 text-[15px] text-ink">{v}</div></div>
                  ))}
                </div>
                <div className="mb-4 h-1 overflow-hidden rounded-full bg-panel-2">
                  <div className="h-full bg-accent transition-[width] duration-100" style={{ width: `${(cur.iteration / cur.total) * 100}%` }} />
                </div>
                <div className="grid gap-6 md:grid-cols-2">
                  <div>
                    <div className="mb-2 flex items-end justify-between">
                      <div className="text-[12.5px] font-medium text-ink-2">U-matrix <span className="font-normal text-ink-3">· mean distance to neighbours</span></div>
                      <ColorRamp scale={uScale} domain={[0, umax]} format={(v) => v.toFixed(1)} width={110} />
                    </div>
                    <SomGrid rows={cur.umatrix.length} cols={cur.umatrix[0].length} width={360} fill={(r, c) => uScale(cur.umatrix[r][c])} ariaLabel="U-matrix" />
                  </div>
                  <div>
                    <div className="mb-2 flex items-end justify-between">
                      <div className="text-[12.5px] font-medium text-ink-2">Hit map <span className="font-normal text-ink-3">· stocks per neuron</span></div>
                      <ColorRamp scale={hScale} domain={[0, hmax]} format={(v) => v.toFixed(0)} width={110} />
                    </div>
                    <SomGrid rows={cur.hits.length} cols={cur.hits[0].length} width={360}
                      fill={(r, c) => hScale(cur.hits[r][c])}
                      renderCell={(r, c, b) => cur.hits[r][c] > 0 && (
                        <text x={b.x + b.w / 2} y={b.y + b.h / 2 + 4} textAnchor="middle" fontSize={11} className="num"
                          style={{ fill: inkOn(hScale(cur.hits[r][c])) }}>{cur.hits[r][c]}</text>
                      )} ariaLabel="Hit map" />
                  </div>
                </div>
              </>
            )}
          </Card>

          {visible.length > 1 && (
            <Card title="Learning curves" subtitle="Dashed line = switch from ordering to convergence phase. Separate axes because the four quantities have different units.">
              <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
                <MiniLine data={visible} dataKey="qe" label="Quantization error (z-units)" color={s[0]} format={(v) => v.toFixed(2)} phaseAt={p.ordering_iters} />
                <MiniLine data={visible} dataKey="te" label="Topographic error (% of stocks)" color={s[1]} format={(v) => `${(v * 100).toFixed(1)}%`} phaseAt={p.ordering_iters} domain={[0, 'auto' as unknown as number]} />
                <MiniLine data={visible} dataKey="eta" label="Learning rate η(t)" color={s[2]} format={(v) => v.toFixed(2)} phaseAt={p.ordering_iters} />
                <MiniLine data={visible} dataKey="sigma" label="Neighbourhood radius σ(t) (cells)" color={s[6]} format={(v) => v.toFixed(1)} phaseAt={p.ordering_iters} />
              </div>
            </Card>
          )}

          {ae && (
            <Card title="Autoencoder training (Units 4–5)" subtitle={`7 → 16 → 3 → 16 → 7 · He init, dropout, L2, Adam · early stopping kept epoch ${ae.best_epoch} of ${ae.epochs_run}. Training loss sits above validation loss because dropout is only active during training.`}>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={ae.train_loss.map((t, i) => ({ epoch: i + 1, train: t, val: ae.val_loss[i] }))} margin={{ top: 4, right: 12, bottom: 14, left: 4 }}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="epoch" type="number" tick={tick} label={axisLabel('epoch')} />
                  <YAxis tick={tick} width={50} tickFormatter={(v) => v.toFixed(2)} label={axisLabel('MSE loss', -90)} />
                  <Tooltip {...tooltip} labelFormatter={(v) => `epoch ${v}`} formatter={(v, n) => [Number(v).toFixed(4), n]} />
                  <Legend {...legend} />
                  <ReferenceLine x={ae.best_epoch} stroke={s[3]} strokeDasharray="3 3" label={{ value: 'best', fill: s[3], fontSize: 11, position: 'top' }} />
                  <Line dataKey="train" name="Training loss" stroke={s[0]} strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line dataKey="val" name="Validation loss" stroke={s[1]} strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

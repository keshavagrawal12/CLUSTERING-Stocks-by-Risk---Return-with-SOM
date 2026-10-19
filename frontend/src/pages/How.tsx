import * as d3 from 'd3'
import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { api } from '../api/client'
import type { StepData } from '../api/types'
import { SomGrid } from '../components/SomGrid'
import { Button, Card, ColorRamp, ErrorBox, PageHeader, Segmented, Spinner, TipRow, useTooltip } from '../components/ui'
import { FEATURE_SHORT, shortTicker } from '../lib/format'
import { INK, inkOn, sequential } from '../lib/palette'
import { useApp } from '../lib/store'

const ROWS = 7, COLS = 7

export default function HowPage() {
  return (
    <div>
      <PageHeader eyebrow="Viva" title="How it works">
        The ideas behind every model in this project, with a live single-step demo of the Kohonen learning rule run by
        the same backend code that trains the real map.
      </PageHeader>
      <div className="space-y-5">
        <NeuronCard />
        <StepDemo />
        <Taxonomy />
        <div className="grid gap-5 lg:grid-cols-2">
          <Card title="Quality metrics">
            <dl className="space-y-4 text-[13.5px] leading-relaxed">
              <div><dt className="font-semibold text-ink">Quantization error (QE)</dt>
                <dd className="text-ink-2">The average distance between each stock and its best-matching neuron's prototype: how well the neurons <em>represent</em> the data.</dd></div>
              <div><dt className="font-semibold text-ink">Topographic error (TE)</dt>
                <dd className="text-ink-2">The share of stocks whose best and second-best neurons are not grid neighbours: how well the map <em>preserves topology</em>.</dd></div>
            </dl>
          </Card>
          <Card title="The training loop (backend/app/models/som.py)">
            <pre className="num scroll-thin overflow-x-auto rounded-lg bg-panel-2 p-4 text-[12px] leading-relaxed text-ink-2">{`for t in range(total):
    phase, eta, sigma = self.schedule(t)   # exp. decay
    x = X[rng.integers(len(X))]            # one random stock
    winner = self.bmu(x)                   # 1. competition
    self.update(x, winner, eta, sigma)     # 2. cooperation
                                           # 3. adaptation:
#   h_j = exp(-d_grid(j, bmu)^2 / (2 sigma^2))
#   w_j <- w_j + eta * h_j * (x - w_j)`}</pre>
          </Card>
        </div>
        <Syllabus />
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ neuron diagram
function NeuronCard() {
  const { theme, stocks } = useApp()
  const ink = INK[theme]
  const feats = stocks?.features ?? ['Annual_Return', 'Volatility', 'Beta', 'Correlation', 'Max_Drawdown', 'Sharpe_Ratio', 'Log_Traded_Value']
  const W = 560, H = 250
  return (
    <Card title="The artificial neuron in a SOM (Unit 1)">
      <div className="grid items-center gap-6 lg:grid-cols-[1.1fr_1fr]">
        <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full">
          {feats.map((f, i) => {
            const y = 22 + i * 32
            return (
              <g key={f}>
                <text x={86} y={y + 4} textAnchor="end" fontSize={12} style={{ fill: ink.text2 }}>{FEATURE_SHORT[f]}</text>
                <circle cx={100} cy={y} r={7} fill="none" stroke={ink.text3} strokeWidth={1.4} />
                <text x={100} y={y + 3.5} textAnchor="middle" fontSize={8.5} className="num" style={{ fill: ink.text3 }}>x{i + 1}</text>
                <line x1={108} y1={y} x2={318} y2={H / 2} stroke={ink.border} strokeWidth={1.4} />
                <text x={108 + 210 * 0.32} y={y + (H / 2 - y) * 0.32 - 4} fontSize={10} className="num" style={{ fill: ink.accent }}>w{i + 1}</text>
              </g>
            )
          })}
          <circle cx={350} cy={H / 2} r={34} fill={ink.panel} stroke={ink.accent} strokeWidth={2} />
          <text x={350} y={H / 2 - 4} textAnchor="middle" fontSize={13} style={{ fill: ink.text }}>neuron j</text>
          <text x={350} y={H / 2 + 13} textAnchor="middle" fontSize={11} className="num" style={{ fill: ink.text3 }}>w_j ∈ ℝ⁷</text>
          <line x1={384} y1={H / 2} x2={440} y2={H / 2} stroke={ink.text2} strokeWidth={1.6} markerEnd="url(#arr)" />
          <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill={ink.text2} /></marker></defs>
          <text x={448} y={H / 2 - 6} fontSize={13} className="num" style={{ fill: ink.text }}>d_j = ‖x − w_j‖</text>
          <text x={448} y={H / 2 + 14} fontSize={11} style={{ fill: ink.text3 }}>output = distance</text>
        </svg>
        <div className="space-y-3 text-[13.5px] leading-relaxed text-ink-2">
          <p>Each of the 49 neurons is an artificial neuron with one weight per input feature. Unlike a perceptron it doesn't compute
            <span className="num text-ink"> f(Σ wᵢxᵢ + b)</span>. Its output is the <span className="text-ink">Euclidean distance</span> between the input and its weights.</p>
          <p>So the weight vector is a <span className="text-ink">prototype</span>: a typical stock behaviour that this neuron "stands for". The neuron
            with the smallest distance wins (the BMU), and training moves prototypes towards the stocks they win.</p>
          <p className="text-[12.5px] text-ink-3">Inputs are z-scored, so every feature has mean 0 and std 1 and gets an equal vote in the distance.</p>
        </div>
      </div>
    </Card>
  )
}

// ------------------------------------------------------------------ single-step demo
function StepDemo() {
  const { stocks, status, theme } = useApp()
  const tip = useTooltip()
  const [ticker, setTicker] = useState<string>('')
  const [eta, setEta] = useState(0.5)
  const [sigma, setSigma] = useState(1.5)
  const [init, setInit] = useState<'random' | 'pca'>('random')
  const [weights, setWeights] = useState<number[][][] | null>(null)
  const [preview, setPreview] = useState<StepData | null>(null)
  const [focus, setFocus] = useState<[number, number] | null>(null)
  const [steps, setSteps] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { if (!ticker && stocks?.stocks.length) setTicker(stocks.stocks[0].ticker) }, [stocks, ticker])

  const compute = useCallback(async (w: number[][][] | null, t: string) => {
    if (!t) return null
    setBusy(true); setError(null)
    try {
      const r = await api.step({ ticker: t, weights: w, rows: ROWS, cols: COLS, eta, sigma, init, seed: 42 })
      setPreview(r)
      if (!w) setWeights(r.weights_before)
      return r
    } catch (e) { setError((e as Error).message); return null } finally { setBusy(false) }
  }, [eta, sigma, init])

  // Recompute the preview when the stock, η or σ changes (weights unchanged).
  useEffect(() => { if (ticker && status?.loaded) void compute(weights, ticker) }, [ticker, eta, sigma]) // eslint-disable-line react-hooks/exhaustive-deps
  // Changing the initialisation resets the map.
  useEffect(() => { setWeights(null); setSteps(0); if (ticker && status?.loaded) void compute(null, ticker) }, [init]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!status?.loaded || !stocks) {
    return <Card title="Interactive single step"><p className="text-[13px] text-ink-3">Load data first to run the demo on real stocks.</p></Card>
  }

  const apply = async () => {
    if (!preview) return
    const w = preview.weights_after
    setWeights(w); setSteps((s) => s + 1)
    await compute(w, ticker)
  }
  const randomSteps = async (n: number) => {
    let w = weights
    for (let i = 0; i < n; i++) {
      const t = stocks.stocks[Math.floor(Math.random() * stocks.stocks.length)].ticker
      const r = await api.step({ ticker: t, weights: w, rows: ROWS, cols: COLS, eta, sigma, init, seed: 42 }).catch(() => null)
      if (!r) break
      w = r.weights_after
    }
    setWeights(w); setSteps((s) => s + n)
    await compute(w, ticker)
  }
  const reset = () => { setWeights(null); setSteps(0); setFocus(null); void compute(null, ticker) }

  const p = preview
  const j = focus ?? p?.bmu ?? [0, 0]
  const dMax = p ? d3.max(p.distances.flat()) ?? 1 : 1
  const dScale = sequential(theme, [dMax, 0])   // close = dark/bright end -> BMU stands out
  const hScale = sequential(theme, [0, 1])

  return (
    <Card title="Interactive single step: the Kohonen rule, one stock at a time"
      subtitle="Pick a stock, look at which neuron wins and how far the neighbourhood reaches, then press Apply to move the weights. Click any neuron to see its numbers in the formula.">
      <div className="mb-5 flex flex-wrap items-end gap-5">
        <label className="text-[12.5px] text-ink-2">
          <div className="mb-1 font-medium">Input stock x</div>
          <select value={ticker} onChange={(e) => setTicker(e.target.value)}
            className="num rounded-md border border-line bg-panel-2 px-2 py-1.5 text-[12.5px] text-ink outline-none focus:border-accent">
            {stocks.stocks.map((s) => <option key={s.ticker} value={s.ticker}>{shortTicker(s.ticker)} · {s.industry}</option>)}
          </select>
        </label>
        <label className="text-[12.5px] text-ink-2">
          <div className="mb-1 font-medium">Learning rate η = <span className="num text-ink">{eta.toFixed(2)}</span></div>
          <input type="range" min={0.01} max={1} step={0.01} value={eta} onChange={(e) => setEta(Number(e.target.value))} className="w-40" />
        </label>
        <label className="text-[12.5px] text-ink-2">
          <div className="mb-1 font-medium">Radius σ = <span className="num text-ink">{sigma.toFixed(2)}</span> cells</div>
          <input type="range" min={0.3} max={4} step={0.05} value={sigma} onChange={(e) => setSigma(Number(e.target.value))} className="w-40" />
        </label>
        <div className="text-[12.5px] text-ink-2">
          <div className="mb-1 font-medium">Start from</div>
          <Segmented value={init} onChange={setInit} options={[{ value: 'random', label: 'Random' }, { value: 'pca', label: 'PCA' }]} />
        </div>
        <div className="flex flex-wrap gap-2">
          <Button onClick={apply} disabled={!p || busy}>{busy && <Spinner />}Apply step</Button>
          <Button variant="ghost" onClick={() => randomSteps(25)} disabled={busy}>+25 random steps</Button>
          <Button variant="ghost" onClick={reset} disabled={busy}>Reset map</Button>
        </div>
        <div className="num text-[12px] text-ink-3">steps applied: <span className="text-ink">{steps}</span></div>
      </div>
      {error && <div className="mb-4"><ErrorBox message={error} /></div>}

      {p && (
        <div className="grid gap-6 xl:grid-cols-[1fr_1fr_1.25fr]">
          <div>
            <div className="mb-1 text-[12.5px] font-medium text-ink-2">1 · Competition: distance of every neuron to x</div>
            <div className="mb-2 text-[11.5px] text-ink-3">Brightest = closest. The ringed neuron is the BMU (winner-takes-all).</div>
            <SomGrid rows={ROWS} cols={COLS} width={300} ring={p.bmu} highlight={focus ? [focus] : []}
              fill={(r, c) => dScale(p.distances[r][c])}
              renderCell={(r, c, b) => (
                <text x={b.x + b.w / 2} y={b.y + b.h / 2 + 3.5} textAnchor="middle" fontSize={9.5} className="num" style={{ fill: inkOn(dScale(p.distances[r][c])) }}>
                  {p.distances[r][c].toFixed(1)}</text>
              )}
              onClick={(r, c) => setFocus([r, c])}
              onHover={(r, c, e) => tip.show(e, <TipRow label={`‖x − w(${r},${c})‖`} value={p.distances[r][c].toFixed(3)} />)} onLeave={tip.hide} />
            <div className="mt-2"><ColorRamp scale={dScale} domain={[dMax, 0]} format={(v) => v.toFixed(1)} label="distance (z-units), far → near" width={200} /></div>
          </div>
          <div>
            <div className="mb-1 text-[12.5px] font-medium text-ink-2">2 · Cooperation: Gaussian neighbourhood h_j</div>
            <div className="mb-2 text-[11.5px] text-ink-3">h = exp(−d²/2σ²) on the grid. h = 1 at the BMU, fading with grid distance.</div>
            <SomGrid rows={ROWS} cols={COLS} width={300} ring={p.bmu} highlight={focus ? [focus] : []}
              fill={(r, c) => hScale(p.neighbourhood[r][c])}
              renderCell={(r, c, b) => p.neighbourhood[r][c] > 0.005 && (
                <text x={b.x + b.w / 2} y={b.y + b.h / 2 + 3.5} textAnchor="middle" fontSize={9.5} className="num" style={{ fill: inkOn(hScale(p.neighbourhood[r][c])) }}>
                  {p.neighbourhood[r][c].toFixed(2)}</text>
              )}
              onClick={(r, c) => setFocus([r, c])}
              onHover={(r, c, e) => tip.show(e, <TipRow label={`h(${r},${c})`} value={p.neighbourhood[r][c].toFixed(4)} />)} onLeave={tip.hide} />
            <div className="mt-2"><ColorRamp scale={hScale} domain={[0, 1]} format={(v) => v.toFixed(1)} label="neighbourhood strength h" width={200} /></div>
          </div>
          <Formula p={p} j={j as [number, number]} />
        </div>
      )}
    </Card>
  )
}

function Formula({ p, j }: { p: StepData; j: [number, number] }) {
  const [r, c] = j
  const h = p.neighbourhood[r][c]
  const w0 = p.weights_before[r][c]
  const w1 = p.weights_after[r][c]
  const isBmu = r === p.bmu[0] && c === p.bmu[1]
  const dGrid = Math.hypot(r - p.bmu[0], c - p.bmu[1])
  const before = Math.hypot(...p.x.map((x, i) => x - w0[i]))
  const after = Math.hypot(...p.x.map((x, i) => x - w1[i]))
  return (
    <div>
      <div className="mb-1 text-[12.5px] font-medium text-ink-2">3 · Adaptation: Kohonen update for neuron ({r}, {c}){isBmu ? ' · the BMU' : ''}</div>
      <div className="mb-3 rounded-lg bg-panel-2 px-3 py-2.5">
        <div className="num text-[13px] text-ink">w_j ← w_j + η · h_j · (x − w_j)</div>
        <div className="num mt-1.5 text-[12px] text-ink-2">
          h_j = exp(−{dGrid.toFixed(2)}² / (2·{p.sigma.toFixed(2)}²)) = <span className="text-ink">{h.toFixed(4)}</span>
        </div>
        <div className="num mt-0.5 text-[12px] text-ink-2">
          η · h_j = {p.eta.toFixed(2)} × {h.toFixed(4)} = <span className="text-accent">{(p.eta * h).toFixed(4)}</span>
        </div>
      </div>
      <table className="num w-full text-[11.5px]">
        <thead className="text-ink-3"><tr>
          <th className="pb-1 text-left font-sans font-medium">feature</th>
          <th className="pb-1 text-right font-medium">x</th>
          <th className="pb-1 text-right font-medium">w_j</th>
          <th className="pb-1 text-right font-medium">x − w_j</th>
          <th className="pb-1 text-right font-medium">Δw</th>
          <th className="pb-1 text-right font-medium">new w_j</th>
        </tr></thead>
        <tbody>
          {p.features.map((f, i) => (
            <tr key={f} className="border-t border-line/60">
              <td className="py-1 font-sans text-ink-2">{FEATURE_SHORT[f]}</td>
              <td className="py-1 text-right text-ink-2">{p.x[i].toFixed(2)}</td>
              <td className="py-1 text-right text-ink-2">{w0[i].toFixed(2)}</td>
              <td className="py-1 text-right text-ink-3">{(p.x[i] - w0[i]).toFixed(2)}</td>
              <td className="py-1 text-right text-accent">{(w1[i] - w0[i] >= 0 ? '+' : '') + (w1[i] - w0[i]).toFixed(3)}</td>
              <td className="py-1 text-right text-ink">{w1[i].toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-3 text-[12px] text-ink-3">
        Distance to x: <span className="num text-ink-2">{before.toFixed(3)}</span> → <span className="num text-ink">{after.toFixed(3)}</span>
        {' '}(shrinks by the factor 1 − η·h = <span className="num">{(1 - p.eta * h).toFixed(3)}</span>). All values in z-units.
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ taxonomy
function Taxonomy() {
  const { theme } = useApp()
  const ink = INK[theme]
  const box = (x: number, y: number, w: number, title: string, sub: string, hl = false): ReactNode => (
    <g>
      <rect x={x} y={y} width={w} height={46} rx={8} fill={hl ? ink.panel : 'none'} stroke={hl ? ink.accent : ink.border} strokeWidth={hl ? 1.6 : 1.2} />
      <text x={x + w / 2} y={y + 20} textAnchor="middle" fontSize={12.5} style={{ fill: ink.text, fontWeight: 600 }}>{title}</text>
      <text x={x + w / 2} y={y + 36} textAnchor="middle" fontSize={10.5} style={{ fill: ink.text3 }}>{sub}</text>
    </g>
  )
  const link = (x1: number, y1: number, x2: number, y2: number) => <path d={`M${x1},${y1} C${x1},${(y1 + y2) / 2} ${x2},${(y1 + y2) / 2} ${x2},${y2}`} fill="none" stroke={ink.border} strokeWidth={1.3} />
  return (
    <Card title="Where each model sits (Unit 1: taxonomy and learning methods)">
      <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
        <svg viewBox="0 0 520 220" className="h-auto w-full">
          {link(260, 50, 130, 95)}{link(260, 50, 390, 95)}{link(130, 141, 70, 170)}{link(130, 141, 200, 170)}{link(390, 141, 390, 170)}
          {box(170, 4, 180, 'Neural networks', 'Unit 1 taxonomy')}
          {box(40, 95, 180, 'Competitive', 'winner-takes-all layer')}
          {box(300, 95, 180, 'Feedforward', 'trained by backprop')}
          {box(4, 170, 130, 'SOM', 'unsupervised', true)}
          {box(144, 170, 120, 'LVQ1', 'supervised', true)}
          {box(316, 170, 150, 'Autoencoder', 'self-supervised', true)}
        </svg>
        <div className="scroll-thin overflow-x-auto">
          <table className="w-full text-[12.5px]">
            <thead className="text-[11.5px] text-ink-3"><tr className="border-b border-line">
              <th className="py-1.5 pr-3 text-left font-medium">Model</th><th className="py-1.5 pr-3 text-left font-medium">Learning</th>
              <th className="py-1.5 pr-3 text-left font-medium">Rule</th><th className="py-1.5 text-left font-medium">Role here</th>
            </tr></thead>
            <tbody className="text-ink-2">
              <tr className="border-b border-line/60"><td className="py-2 pr-3 font-semibold text-ink">SOM</td><td className="pr-3">Unsupervised competitive</td><td className="num pr-3 text-[11.5px]">w += η·h·(x−w)</td><td>Groups stocks by behaviour (RQ1–3)</td></tr>
              <tr className="border-b border-line/60"><td className="py-2 pr-3 font-semibold text-ink">LVQ1</td><td className="pr-3">Supervised competitive</td><td className="num pr-3 text-[11.5px]">w ±= η·(x−w)</td><td>Predicts sector from behaviour (RQ1)</td></tr>
              <tr><td className="py-2 pr-3 font-semibold text-ink">Autoencoder</td><td className="pr-3">Self-supervised, backprop</td><td className="num pr-3 text-[11.5px]">min ‖x − x̂‖²</td><td>Compresses 7 → 3 features for the SOM</td></tr>
            </tbody>
          </table>
          <p className="mt-3 text-[12.5px] leading-relaxed text-ink-3">SOM and LVQ share the competition step (find the closest prototype). The SOM also lets neighbours learn and never sees labels;
            LVQ uses the label to decide whether to pull the winner closer or push it away. The autoencoder learns by gradient descent on its own reconstruction error.</p>
        </div>
      </div>
    </Card>
  )
}

function Syllabus() {
  const rows = [
    ['Unit 1', 'Model of artificial neuron, learning methods, taxonomy of NN architectures', 'SOM neuron as prototype (diagram above); taxonomy panel (unsupervised / supervised / self-supervised)'],
    ['Unit 3', 'Neural networks based on competition, Kohonen SOM (Practical 8)', 'models/som.py: BMU, neighbourhood, Kohonen rule, decay, QE/TE, U-matrix'],
    ['Unit 3', 'Learning Vector Quantization', 'models/lvq.py: sector prediction from behaviour (RQ1)'],
    ['Unit 4', 'Regularization, dropout, initialization, gradient descent strategies', 'Autoencoder training (He init, dropout, L2, Adam, early stopping)'],
    ['Unit 5', 'Autoencoders (Practicals 13–14)', 'models/autoencoder.py: 3-D latent features fed to the SOM'],
  ]
  return (
    <Card title="Syllabus mapping (BDA404-5N)" pad={false}>
      <table className="w-full text-[13px]">
        <thead className="text-[11.5px] text-ink-3"><tr className="border-b border-line">
          <th className="px-5 py-2 text-left font-medium">Unit</th><th className="px-3 py-2 text-left font-medium">Syllabus topic</th><th className="px-5 py-2 text-left font-medium">Where it appears</th>
        </tr></thead>
        <tbody>
          {rows.map(([u, t, w], i) => (
            <tr key={i} className="border-b border-line/60 last:border-0">
              <td className="num whitespace-nowrap px-5 py-2.5 text-accent">{u}</td>
              <td className="px-3 py-2.5 text-ink">{t}</td>
              <td className="px-5 py-2.5 text-ink-2">{w}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}

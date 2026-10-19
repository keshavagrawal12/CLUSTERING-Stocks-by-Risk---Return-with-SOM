import { CartesianGrid, Line, LineChart, ReferenceDot, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api/client'
import type { RQ1Data, Variant } from '../api/types'
import { Heatmap, useChartStyle } from '../components/charts'
import { Badge, Card, ErrorBox, Loading, PageHeader, Requires, SectorDot, Stat, TipRow } from '../components/ui'
import { FEATURE_LABEL, fmtFeature, pct } from '../lib/format'
import { SERIES } from '../lib/palette'
import { useApp, useFetch } from '../lib/store'

export default function ClustersPage() {
  return <Requires run>{(runId) => <ClustersView runId={runId} />}</Requires>
}

function ClustersView({ runId }: { runId: string }) {
  const rq1 = useFetch(`rq1-${runId}`, () => api.rq1(runId))
  const cl = useFetch(`clusters-${runId}`, () => api.clusters(runId))
  if (rq1.error) return <ErrorBox message={rq1.error} />
  if (!rq1.data) return <Loading tall label="Computing RQ1 (first time trains the comparison maps and the autoencoder)…" />
  const d = rq1.data
  return (
    <div>
      <PageHeader eyebrow="Findings · RQ1" title="Do behaviour clusters match official sectors?">
        <span className="text-ink">{d.headline}.</span> ARI compares every pair of stocks: do the clusters and the sectors agree
        on "same group" vs "different group"? 0 = no better than chance, 1 = identical. NMI measures how much the cluster tells you about the sector.
      </PageHeader>

      <div className="mb-5 grid gap-4 md:grid-cols-3">
        {d.variants.map((v) => <VariantCard key={v.key} v={v} current={(d.input_mode === 'raw' && v.key === 'som_raw') || (d.input_mode !== 'raw' && v.key === 'som_ae')} />)}
      </div>

      <div className="mb-5 grid gap-5 lg:grid-cols-[1fr_1.6fr]">
        <Silhouette d={d} />
        <Card title="Clusters × industries" subtitle="Number of stocks in each combination. A perfect sector match would put each industry in a single row.">
          <Heatmap rowLabels={d.contingency.clusters.map((c) => `Cluster C${c}`)} colLabels={d.contingency.industries}
            values={d.contingency.counts} rowTitle="SOM cluster" colTitle="industry"
            cellTip={(r, c) => (
              <div><div className="mb-1 font-semibold">C{d.contingency.clusters[r]} × {d.contingency.industries[c]}</div>
                <TipRow label="Stocks" value={d.contingency.counts[r][c]} />
                <TipRow label="Share of industry" value={pct(d.contingency.counts[r][c] / d.contingency.counts.reduce((a, row) => a + row[c], 0), 0)} /></div>
            )} />
        </Card>
      </div>

      <LVQCard d={d} />

      <Card className="mt-5" title="Cluster profiles" subtitle="Mean of each feature over the stocks in the cluster (real units), with its largest industries"
        pad={false}>
        {cl.error ? <div className="p-5"><ErrorBox message={cl.error} /></div> : !cl.data ? <Loading /> : (
          <div className="scroll-thin overflow-x-auto">
            <table className="w-full text-[12.5px]">
              <thead className="text-[11.5px] text-ink-3">
                <tr className="border-b border-line">
                  <th className="px-4 py-2 text-left font-medium">Cluster</th>
                  <th className="px-3 py-2 text-right font-medium">Stocks</th>
                  {Object.keys(cl.data.profiles[0]?.mean ?? {}).map((f) => <th key={f} className="whitespace-nowrap px-3 py-2 text-right font-medium">{FEATURE_LABEL[f]}</th>)}
                  <th className="px-4 py-2 text-left font-medium">Top industries</th>
                </tr>
              </thead>
              <tbody>
                {cl.data.profiles.map((p) => (
                  <tr key={p.cluster} className="border-b border-line/60">
                    <td className="num px-4 py-2 font-medium text-ink">C{p.cluster}</td>
                    <td className="num px-3 py-2 text-right text-ink-2">{p.size}</td>
                    {Object.entries(p.mean ?? {}).map(([f, v]) => <td key={f} className="num px-3 py-2 text-right text-ink-2">{fmtFeature(f, v)}</td>)}
                    <td className="px-4 py-2">
                      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11.5px] text-ink-2">
                        {p.industries.slice(0, 3).map((i) => <span key={i.industry} className="flex items-center gap-1"><SectorDot industry={i.industry} size={6} />{i.industry} <span className="num text-ink-3">{i.count}</span></span>)}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}

function VariantCard({ v, current }: { v: Variant; current: boolean }) {
  return (
    <div className={`rounded-xl border bg-panel p-5 ${current ? 'border-accent/50' : 'border-line'}`}>
      <div className="mb-4 flex items-center justify-between">
        <span className="text-[13.5px] font-semibold text-ink">{v.name}</span>
        {current ? <Badge tone="accent">this run</Badge> : v.key === 'kmeans' ? <Badge>baseline</Badge> : null}
      </div>
      {!v.available ? <p className="text-[12.5px] text-ink-3">Not available: {v.reason}</p> : (
        <>
          <div className="grid grid-cols-2 gap-4">
            <Stat label="ARI" value={v.ari!.toFixed(3)} tone={current ? 'accent' : undefined} />
            <Stat label="NMI" value={v.nmi!.toFixed(3)} />
          </div>
          <div className="mt-4 flex flex-wrap gap-x-4 gap-y-1 border-t border-line pt-3 text-[12px] text-ink-3">
            <span>k = <span className="num text-ink-2">{v.k}</span></span>
            {v.qe != null && <span>QE <span className="num text-ink-2">{v.qe.toFixed(3)}</span></span>}
            {v.te != null && <span>TE <span className="num text-ink-2">{(v.te * 100).toFixed(1)}%</span></span>}
            {v.ae_reconstruction_mse != null && <span>AE MSE <span className="num text-ink-2">{v.ae_reconstruction_mse.toFixed(3)}</span></span>}
          </div>
          {v.key === 'som_ae' && <p className="mt-2 text-[11px] text-ink-3">QE is in the 3-D latent space, so it is not directly comparable with the 7-D QE.</p>}
        </>
      )}
    </div>
  )
}

function Silhouette({ d }: { d: RQ1Data }) {
  const { tick, tooltip, axisLabel, ink } = useChartStyle()
  const { theme } = useApp()
  const best = d.silhouette.find((s) => s.k === d.k)
  return (
    <Card title="Choosing k" subtitle="K-means is run on the trained neuron weights for k = 3…10; the k with the highest silhouette wins.">
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={d.silhouette} margin={{ top: 16, right: 16, bottom: 16, left: 4 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="k" tick={tick} label={axisLabel('number of clusters k')} />
          <YAxis tick={tick} width={52} domain={['auto', 'auto']} tickFormatter={(v) => v.toFixed(2)} label={axisLabel('silhouette score', -90)} />
          <Tooltip {...tooltip} labelFormatter={(k) => `k = ${k}`} formatter={(v) => [Number(v).toFixed(4), 'silhouette']} />
          <ReferenceLine x={d.k} stroke={ink.accent} strokeDasharray="3 3" />
          <Line dataKey="score" stroke={SERIES[theme][0]} strokeWidth={2} dot={{ r: 4, strokeWidth: 0, fill: SERIES[theme][0] }} isAnimationActive={false} />
          {best && <ReferenceDot x={best.k} y={best.score} r={7} fill={ink.accent} stroke={ink.panel} strokeWidth={2}
            label={{ value: `chosen k = ${d.k}`, position: 'top', fill: ink.accent, fontSize: 11.5 }} />}
        </LineChart>
      </ResponsiveContainer>
    </Card>
  )
}

function LVQCard({ d }: { d: RQ1Data }) {
  const l = d.lvq
  const { theme } = useApp()
  const s = SERIES[theme]
  const lift = l.accuracy - l.baseline_accuracy
  return (
    <Card title="LVQ1: can a supervised competitive network predict the sector from behaviour?"
      subtitle={`${l.folds}-fold stratified cross-validation on ${l.n_samples} stocks · one prototype per sector · sectors with < 3 stocks merged into "Other"`}>
      <div className="grid gap-6 lg:grid-cols-[300px_1fr]">
        <div className="space-y-5">
          <div className="grid grid-cols-2 gap-4">
            <Stat label="LVQ accuracy" value={pct(l.accuracy, 1)} sub={`± ${pct(l.accuracy_std, 1)} across folds`} tone="accent" />
            <Stat label="Majority baseline" value={pct(l.baseline_accuracy, 1)} sub={`always "${l.baseline_class}"`} />
          </div>
          <div className="space-y-2">
            {[{ label: 'LVQ1', v: l.accuracy, c: s[0] }, { label: 'Baseline', v: l.baseline_accuracy, c: s[1] }].map((b) => (
              <div key={b.label} className="flex items-center gap-3 text-[12px]">
                <span className="w-16 text-ink-2">{b.label}</span>
                <div className="h-3 flex-1 rounded-sm bg-panel-2"><div className="h-full rounded-sm" style={{ width: `${b.v * 100}%`, background: b.c }} /></div>
                <span className="num w-12 text-right text-ink">{pct(b.v, 0)}</span>
              </div>
            ))}
            <div className="text-[11px] text-ink-3">Axis: share of stocks whose sector was predicted correctly (0–100%).</div>
          </div>
          <p className="text-[13px] leading-relaxed text-ink-2">
            {lift > 0.05
              ? <>LVQ beats the baseline by <span className="num text-ink">{(lift * 100).toFixed(1)}</span> points: behaviour carries <em>some</em> sector information, but most stocks are still misclassified.</>
              : lift > 0 ? <>LVQ is only <span className="num text-ink">{(lift * 100).toFixed(1)}</span> points above the baseline: behaviour says little about sector.</>
                : <>LVQ does not beat the majority baseline: behaviour alone does not identify the sector.</>}
          </p>
          {l.merged_into_other.length > 0 && <p className="text-[11.5px] leading-snug text-ink-3">Merged into "Other": {l.merged_into_other.join(', ')}.</p>}
        </div>
        <div>
          <div className="mb-2 text-[12.5px] font-medium text-ink-2">Confusion matrix (summed over folds) · outlined diagonal = correct</div>
          <Heatmap rowLabels={l.classes} colLabels={l.classes} values={l.confusion} rowTitle="true sector" colTitle="predicted" highlightDiagonal
            cellTip={(r, c) => (
              <div><div className="mb-1 font-semibold">{l.classes[r]}</div>
                <TipRow label={`predicted ${l.classes[c]}`} value={l.confusion[r][c]} />
                <TipRow label="of this sector" value={pct(l.confusion[r][c] / l.confusion[r].reduce((a, b) => a + b, 0), 0)} /></div>
            )} />
        </div>
      </div>
    </Card>
  )
}

import type { ReactNode } from 'react'
import { api } from '../api/client'
import { Badge, Button, Card, Loading, Spinner } from '../components/ui'
import { navigate, type PageId } from '../lib/router'
import { useApp, useFetch } from '../lib/store'
import { pct, pValue, shortTicker } from '../lib/format'

function RQCard({ tag, question, page, children, loading, error }: {
  tag: string; question: string; page: PageId; children?: ReactNode; loading?: boolean; error?: string | null
}) {
  return (
    <button onClick={() => navigate(page)}
      className="group flex flex-col rounded-xl border border-line bg-panel p-5 text-left transition-colors hover:border-line-strong">
      <div className="flex items-center justify-between">
        <span className="num text-[12px] font-medium text-accent">{tag}</span>
        <span className="text-[12px] text-ink-3 transition-colors group-hover:text-ink-2">Open →</span>
      </div>
      <p className="mt-3 text-[15px] font-medium leading-snug text-ink">{question}</p>
      <div className="mt-4 min-h-[64px] border-t border-line pt-4 text-[13.5px] leading-relaxed text-ink-2">
        {loading ? <span className="flex items-center gap-2 text-ink-3"><Spinner /> Computing…</span>
          : error ? <span className="text-critical">{error}</span>
            : children ?? <span className="text-ink-3">Train a map to answer this.</span>}
      </div>
    </button>
  )
}

export default function Overview() {
  const { status, runId, statusError } = useApp()
  const rq1 = useFetch(runId ? `rq1-${runId}` : null, () => api.rq1(runId!))
  const rq2 = useFetch(runId ? `rq2-${runId}` : null, () => api.rq2(runId!))
  const rq3 = useFetch(runId ? `rq3-${runId}` : null, () => api.rq3(runId!))

  return (
    <div>
      <div className="mb-10 max-w-3xl">
        <div className="mb-3 text-[11px] font-medium uppercase tracking-[0.14em] text-accent">BDA404-5N · Course project</div>
        <h1 className="text-[34px] font-semibold leading-[1.15] tracking-tight text-ink">
          Do NIFTY 100 stocks behave like their sector, or is the sector label hiding how they really move?
        </h1>
        <p className="mt-4 text-[15px] leading-relaxed text-ink-2">
          A Kohonen Self-Organizing Map groups stocks by seven risk–return features from 2022–2024. We compare those
          behaviour clusters with the official sectors, find stocks that don't fit, and test the clusters
          out-of-sample on 2025 portfolios.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <RQCard tag="RQ1" page="clusters" question="Do stocks grouped by risk and return behaviour match their official sectors?"
          loading={rq1.loading} error={rq1.error}>
          {rq1.data && (
            <>
              <div className="text-ink">{rq1.data.headline}.</div>
              <div className="mt-2 text-[12.5px] text-ink-3">
                NMI <span className="num text-ink-2">{rq1.data.nmi.toFixed(2)}</span> · LVQ sector accuracy{' '}
                <span className="num text-ink-2">{pct(rq1.data.lvq.accuracy, 0)}</span> vs baseline{' '}
                <span className="num text-ink-2">{pct(rq1.data.lvq.baseline_accuracy, 0)}</span>
              </div>
            </>
          )}
        </RQCard>
        <RQCard tag="RQ2" page="misfits" question="Which stocks behave differently from the rest of their sector?"
          loading={rq2.loading} error={rq2.error}>
          {rq2.data && (
            <>
              <div className="text-ink">{rq2.data.headline}.</div>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {rq2.data.misfits.slice(0, 5).map((m) => <Badge key={m.ticker}><span className="num">{shortTicker(m.ticker)}</span></Badge>)}
              </div>
            </>
          )}
        </RQCard>
        <RQCard tag="RQ3" page="portfolio" question="Does diversifying across SOM clusters lower 2025 risk more than diversifying across sectors?"
          loading={rq3.loading} error={rq3.error}>
          {rq3.data && (
            <>
              <div className="text-ink">{rq3.data.verdict}</div>
              <div className="mt-2 text-[12.5px] text-ink-3">
                Mann–Whitney p = <span className="num text-ink-2">{pValue(rq3.data.mann_whitney.p_two_sided)}</span> ·{' '}
                <span className="num">{rq3.data.n_portfolios.toLocaleString()}</span> portfolios per rule
              </div>
            </>
          )}
        </RQCard>
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-[1.3fr_1fr]">
        <Card title="Method" subtitle="Each step maps to a syllabus unit, see How it works">
          <ol className="grid gap-4 sm:grid-cols-4">
            {[
              ['1', 'Features', '7 behaviour features per stock, 2022–24 only, z-scored'],
              ['2', 'SOM', 'Unsupervised competitive learning on a 7×7 grid (Unit 3)'],
              ['3', 'Clusters', 'K-means over neuron weights, k by silhouette'],
              ['4', 'Test', 'RQ1 agreement · RQ2 misfits · RQ3 2025 portfolios'],
            ].map(([n, t, d]) => (
              <li key={n}>
                <div className="num text-[12px] text-accent">{n.padStart(2, '0')}</div>
                <div className="mt-1 text-[13.5px] font-semibold text-ink">{t}</div>
                <div className="mt-1 text-[12.5px] leading-snug text-ink-3">{d}</div>
              </li>
            ))}
          </ol>
        </Card>
        <Card title="Data status">
          {statusError ? <div className="text-[13px] text-critical">{statusError}</div>
            : !status ? <Loading />
              : status.loaded ? (
                <div className="space-y-3 text-[13px]">
                  <div className="grid grid-cols-3 gap-4">
                    <div><div className="text-[11.5px] uppercase tracking-[0.08em] text-ink-3">Stocks</div><div className="num mt-1 text-[22px] text-ink">{status.n_stocks}</div></div>
                    <div><div className="text-[11.5px] uppercase tracking-[0.08em] text-ink-3">Industries</div><div className="num mt-1 text-[22px] text-ink">{status.n_industries}</div></div>
                    <div><div className="text-[11.5px] uppercase tracking-[0.08em] text-ink-3">Dropped</div><div className="num mt-1 text-[22px] text-ink">{status.dropped.length}</div></div>
                  </div>
                  <div className="text-[12.5px] text-ink-3">Source: <span className="num text-ink-2">{status.source}</span></div>
                  <div className="flex gap-2">
                    {!runId && <Button small onClick={() => navigate('train')}>Train the first map</Button>}
                    <Button small variant="ghost" onClick={() => navigate('data')}>View data</Button>
                  </div>
                </div>
              ) : (
                <div className="text-[13.5px] text-ink-2">
                  No data loaded yet. Upload the three CSVs on the Data page or run the fetch script.
                  <div className="mt-3"><Button small onClick={() => navigate('data')}>Load data</Button></div>
                </div>
              )}
        </Card>
      </div>
    </div>
  )
}

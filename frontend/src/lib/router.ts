// Minimal hash router: #/map, #/train ... (no extra dependency needed for 8 pages).
import { useEffect, useState } from 'react'

export const PAGES = [
  { id: 'overview', label: 'Overview', group: 'Project' },
  { id: 'data', label: 'Data', group: 'Project' },
  { id: 'train', label: 'Train', group: 'Model' },
  { id: 'map', label: 'Map', group: 'Model' },
  { id: 'clusters', label: 'Clusters vs Sectors', group: 'Findings', tag: 'RQ1' },
  { id: 'misfits', label: 'Misfits', group: 'Findings', tag: 'RQ2' },
  { id: 'portfolio', label: 'Portfolio test', group: 'Findings', tag: 'RQ3' },
  { id: 'how', label: 'How it works', group: 'Viva' },
] as const

export type PageId = (typeof PAGES)[number]['id']

function current(): PageId {
  const h = window.location.hash.replace(/^#\/?/, '')
  return (PAGES.find((p) => p.id === h)?.id ?? 'overview') as PageId
}

export function navigate(id: PageId) {
  window.location.hash = `/${id}`
}

export function useRoute(): PageId {
  const [page, setPage] = useState<PageId>(current)
  useEffect(() => {
    const on = () => { setPage(current()); window.scrollTo(0, 0) }
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return page
}

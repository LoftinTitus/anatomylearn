import { Suspense, useCallback, useEffect, useMemo, useState } from 'react'
import { BodyViewer } from './components/BodyViewer'
import { InfoPanel } from './components/InfoPanel'
import { Sidebar } from './components/Sidebar'
import { SliceViewer } from './components/SliceViewer'
import type { Manifest, Vec3 } from './types'

export default function App() {
  const [manifest, setManifest] = useState<Manifest | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [crosshairRas, setCrosshairRas] = useState<Vec3 | null>(null)
  const [hiddenSystems, setHiddenSystems] = useState<Set<string>>(new Set())
  const [xray, setXray] = useState(true)

  useEffect(() => {
    fetch('/data/structures.json')
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`)
        return r.json()
      })
      .then(setManifest)
      .catch(() => setError('No data found. Run the pipeline first (see README).'))
  }, [])

  const byLabel = useMemo(() => new Map(manifest?.structures.map((s) => [s.label, s])), [manifest])
  const selected = manifest?.structures.find((s) => s.id === selectedId) ?? null
  const visible = useMemo(
    () => new Set(manifest?.structures.filter((s) => !hiddenSystems.has(s.system)).map((s) => s.id)),
    [manifest, hiddenSystems],
  )

  const pickFromModel = useCallback((id: string, ras: Vec3) => {
    setSelectedId(id)
    setCrosshairRas(ras)
  }, [])

  // Clicking background on the slices clears the selection but still moves the crosshair.
  const pickFromSlices = useCallback(
    (label: number, ras: Vec3) => {
      setSelectedId(byLabel.get(label)?.id ?? null)
      setCrosshairRas(ras)
    },
    [byLabel],
  )

  const selectFromList = (id: string) => {
    const s = manifest?.structures.find((x) => x.id === id)
    if (!s) return
    setSelectedId(id)
    setCrosshairRas(s.anchorRas)
    setHiddenSystems((h) => {
      if (!h.has(s.system)) return h
      const next = new Set(h)
      next.delete(s.system)
      return next
    })
  }

  const toggleSystem = (system: string) =>
    setHiddenSystems((h) => {
      const next = new Set(h)
      if (!next.delete(system)) next.add(system)
      return next
    })

  if (error) return <div className="splash">{error}</div>
  if (!manifest) return <div className="splash">Loading…</div>

  return (
    <div className="app">
      <header className="topbar">
        <h1>AnatomyLearn</h1>
        <span className="muted">
          {manifest.source} · {manifest.segmentation}
        </span>
        <label className="toggle">
          <input type="checkbox" checked={xray} onChange={(e) => setXray(e.target.checked)} />
          X-ray selected
        </label>
      </header>
      <Sidebar
        structures={manifest.structures}
        hiddenSystems={hiddenSystems}
        selectedId={selectedId}
        onToggleSystem={toggleSystem}
        onSelect={selectFromList}
      />
      <main className="model">
        <Suspense fallback={<div className="splash">Loading model…</div>}>
          <BodyViewer
            manifest={manifest}
            visible={visible}
            selectedId={selectedId}
            crosshairRas={crosshairRas}
            xray={xray}
            onPick={pickFromModel}
          />
        </Suspense>
      </main>
      <div className="right">
        <SliceViewer manifest={manifest} selectedId={selectedId} crosshairRas={crosshairRas} onPick={pickFromSlices} />
        <InfoPanel structure={selected} />
      </div>
    </div>
  )
}

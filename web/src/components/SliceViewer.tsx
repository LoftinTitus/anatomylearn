import { Niivue, NVImage, SHOW_RENDER, SLICE_TYPE } from '@niivue/niivue'
import { useEffect, useRef, useState } from 'react'
import type { Manifest, Vec3 } from '../types'

// [window min, window max] in Hounsfield units.
const WINDOWS = {
  'Soft tissue': [-160, 240],
  Lung: [-1350, 150],
  Bone: [-450, 1050],
} as const
type WindowName = keyof typeof WINDOWS

const VIEWS = {
  Axial: SLICE_TYPE.AXIAL,
  Coronal: SLICE_TYPE.CORONAL,
  Sagittal: SLICE_TYPE.SAGITTAL,
  All: SLICE_TYPE.MULTIPLANAR,
} as const
type ViewName = keyof typeof VIEWS

interface Props {
  manifest: Manifest
  selectedId: string | null
  crosshairRas: Vec3 | null
  onPick: (label: number, ras: Vec3) => void
}

export function SliceViewer({ manifest, selectedId, crosshairRas, onPick }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const nvRef = useRef<Niivue | null>(null)
  const [ready, setReady] = useState(false)
  // A low-res CT shows first; the full-resolution one replaces it when downloaded.
  const [fullCt, setFullCt] = useState(false)
  const [windowName, setWindowName] = useState<WindowName>('Soft tissue')
  const [showLabels, setShowLabels] = useState(true)
  const [view, setView] = useState<ViewName>('Axial')
  const onPickRef = useRef(onPick)
  useEffect(() => {
    onPickRef.current = onPick
  }, [onPick])

  useEffect(() => {
    const nv = new Niivue({
      backColor: [0, 0, 0, 1],
      crosshairColor: [0.22, 0.74, 0.97, 1],
      crosshairGap: 6,
      isRadiologicalConvention: true, // patient's right on screen left, as in clinical viewing
      multiplanarShowRender: SHOW_RENDER.NEVER,
      isOrientCube: false,
      loadingText: 'Loading CT...',
      // Patched option (patches/@niivue+niivue+*.patch): skip the 3D-render gradient pass,
      // which otherwise runs on every refresh and dominates selection latency.
      skipGradients: true,
    } as ConstructorParameters<typeof Niivue>[0])
    nvRef.current = nv
    let cancelled = false
    const ctOptions = { colormap: 'gray', cal_min: WINDOWS['Soft tissue'][0], cal_max: WINDOWS['Soft tissue'][1] }
    // Start the big download right away, in parallel with the preview.
    const full = NVImage.loadFromUrl({ url: '/data/ct.nii.gz', ...ctOptions })
    nv.attachToCanvas(canvasRef.current!)
      .then(() =>
        nv.loadVolumes([
          { url: '/data/ct_preview.nii.gz', ...ctOptions },
          { url: '/data/seg.nii.gz', opacity: 0.45 },
        ]),
      )
      .then(() => {
        if (cancelled) return
        nv.onLocationChange = (loc) => {
          const { mm, values } = loc as { mm: number[]; values: { value: number }[] }
          onPickRef.current(Math.round(values[1]?.value ?? 0), [mm[0], mm[1], mm[2]])
        }
        setReady(true)
        return full
      })
      .then((image) => {
        if (cancelled || !image) return
        const preview = nv.volumes[0]
        nv.addVolume(image)
        nv.setVolume(image, 0) // becomes the background; the label map stays on top
        nv.removeVolume(preview)
        setFullCt(true)
      })
      .catch((err) => console.warn('CT failed to load', err))
    return () => {
      cancelled = true
      nv.cleanup()
      nvRef.current = null
    }
  }, [])

  // Label colours: all structures, or only the selected one. NiiVue renders label
  // alpha as on/off, so unselected labels are hidden rather than dimmed.
  useEffect(() => {
    const nv = nvRef.current
    if (!ready || !nv) return
    const seg = nv.volumes[1]
    // The transparent -1 entry keeps background (0) off the LUT's edge texel: NiiVue's
    // shader nudges the lowest value upward, which otherwise blends in label 1's colour.
    const rows = [
      { label: -1, color: [0, 0, 0], alpha: 0, name: '' },
      { label: 0, color: [0, 0, 0], alpha: 0, name: '' },
    ].concat(
      manifest.structures.map((s) => ({
        label: s.label,
        color: s.color,
        // Body outline labels would wash over the whole body, so show them only when selected.
        alpha: s.id === selectedId || (selectedId === null && !s.shell) ? 255 : 0,
        name: s.name,
      })),
    )
    seg.setColormapLabel({
      R: rows.map((r) => r.color[0]),
      G: rows.map((r) => r.color[1]),
      B: rows.map((r) => r.color[2]),
      A: rows.map((r) => r.alpha),
      I: rows.map((r) => r.label),
      labels: rows.map((r) => r.name),
    })
    nv.updateGLVolume()
  }, [ready, manifest, selectedId])

  // setOpacity refreshes the GPU textures itself, so keep it out of the selection path.
  useEffect(() => {
    if (ready) nvRef.current?.setOpacity(1, showLabels ? 0.55 : 0)
  }, [ready, showLabels])

  useEffect(() => {
    if (ready) nvRef.current?.setSliceType(VIEWS[view])
  }, [ready, view])

  useEffect(() => {
    const nv = nvRef.current
    if (!ready || !nv) return
    const [min, max] = WINDOWS[windowName]
    nv.volumes[0].cal_min = min
    nv.volumes[0].cal_max = max
    nv.updateGLVolume()
  }, [ready, fullCt, windowName])

  // Follow selections made in the 3D view or sidebar (setting crosshairPos does not re-fire onLocationChange).
  useEffect(() => {
    const nv = nvRef.current
    if (!ready || !nv || !crosshairRas) return
    nv.scene.crosshairPos = nv.mm2frac(crosshairRas, 0, true)
    nv.drawScene()
  }, [ready, fullCt, crosshairRas])

  return (
    <div className="slice-viewer">
      <div className="toolbar">
        <div className="segmented" role="group" aria-label="Slice view">
          {(Object.keys(VIEWS) as ViewName[]).map((name) => (
            <button key={name} className={name === view ? 'active' : ''} onClick={() => setView(name)}>
              {name}
            </button>
          ))}
        </div>
        {(Object.keys(WINDOWS) as WindowName[]).map((name) => (
          <button key={name} className={name === windowName ? 'active' : ''} onClick={() => setWindowName(name)}>
            {name}
          </button>
        ))}
        {ready && !fullCt && <span className="loading-badge">Loading full detail…</span>}
        <label className="toggle">
          <input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} />
          Labels
        </label>
      </div>
      <div className="canvas-wrap">
        <canvas ref={canvasRef} />
      </div>
    </div>
  )
}

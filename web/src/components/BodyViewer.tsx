import { Html, OrbitControls, useGLTF } from '@react-three/drei'
import { Canvas, type ThreeEvent } from '@react-three/fiber'
import { useMemo, useState } from 'react'
import * as THREE from 'three'
import { rasToThree, threeToRas } from '../coords'
import type { Manifest, Structure, Vec3 } from '../types'

interface Props {
  manifest: Manifest
  visible: Set<string>
  selectedId: string | null
  crosshairRas: Vec3 | null
  xray: boolean
  onPick: (id: string, ras: Vec3) => void
}

export function BodyViewer(props: Props) {
  const [lo, hi] = props.manifest.volumeBoundsRas
  const a = new THREE.Vector3(...rasToThree(lo))
  const b = new THREE.Vector3(...rasToThree(hi))
  const center = a.clone().add(b).multiplyScalar(0.5)
  const size = a.distanceTo(b)

  return (
    <Canvas camera={{ position: [center.x, center.y, center.z + size * 1.6], fov: 35, near: 1, far: size * 10 }}>
      <color attach="background" args={['#0d1117']} />
      <hemisphereLight args={['#ffffff', '#334155', 1.2]} />
      <directionalLight position={[center.x + size, center.y + size, center.z + size]} intensity={1.6} />
      <directionalLight position={[center.x - size, center.y, center.z - size]} intensity={0.5} />
      <Model {...props} />
      <Crosshair crosshairRas={props.crosshairRas} lo={a} hi={b} />
      <OrbitControls target={center} makeDefault />
    </Canvas>
  )
}

function Model({ manifest, visible, selectedId, xray, onPick }: Props) {
  const { scene } = useGLTF('/data/body.glb')
  const [hover, setHover] = useState<{ id: string; point: THREE.Vector3 } | null>(null)

  // Keep each node's world transform rather than baking it in: the compressed model
  // stores quantized integer positions that the node transform scales back to millimetres.
  const parts = useMemo(() => {
    const byId = new Map(manifest.structures.map((s) => [s.id, s]))
    const out: { structure: Structure; geometry: THREE.BufferGeometry; matrix: THREE.Matrix4 }[] = []
    scene.updateMatrixWorld(true)
    scene.traverse((obj) => {
      if (!(obj instanceof THREE.Mesh)) return
      const structure = byId.get(obj.name) ?? byId.get(obj.parent?.name ?? '')
      if (!structure) return
      out.push({ structure, geometry: obj.geometry, matrix: obj.matrixWorld.clone() })
    })
    return out
  }, [scene, manifest])

  const hovered = hover && manifest.structures.find((s) => s.id === hover.id)

  return (
    <group>
      {parts.map(({ structure, geometry, matrix }) => {
        if (!visible.has(structure.id)) return null
        const isSelected = structure.id === selectedId
        if (structure.shell) {
          // No pointer handlers: R3F only raycasts interactive meshes, so clicks reach the organs inside.
          return (
            <mesh key={structure.id} geometry={geometry} matrix={matrix} matrixAutoUpdate={false} renderOrder={2}>
              <meshStandardMaterial
                color={isSelected ? '#ffd166' : '#d9c2b0'}
                roughness={0.8}
                transparent
                opacity={isSelected ? 0.35 : 0.1}
                depthWrite={false}
                side={THREE.FrontSide}
              />
            </mesh>
          )
        }
        const faded = xray && selectedId !== null && !isSelected
        return (
          <mesh
            key={structure.id}
            geometry={geometry}
            matrix={matrix}
            matrixAutoUpdate={false}
            renderOrder={faded ? 1 : 0}
            onPointerMove={(e: ThreeEvent<PointerEvent>) => {
              e.stopPropagation()
              setHover({ id: structure.id, point: e.point.clone() })
            }}
            onPointerOut={() => setHover((h) => (h?.id === structure.id ? null : h))}
            onClick={(e: ThreeEvent<MouseEvent>) => {
              if (e.delta > 4) return // orbit drag, not a click
              e.stopPropagation()
              onPick(structure.id, threeToRas(e.point.toArray() as Vec3))
            }}
          >
            <meshStandardMaterial
              // three.js bakes opacity mode into the shader; remount when it changes.
              key={faded ? 'faded' : 'solid'}
              color={new THREE.Color(...structure.color.map((c) => c / 255) as Vec3)}
              roughness={0.55}
              metalness={0.05}
              emissive={isSelected ? '#ffd166' : hover?.id === structure.id ? '#555555' : '#000000'}
              emissiveIntensity={isSelected ? 0.45 : 0.4}
              transparent={faded}
              opacity={faded ? 0.12 : 1}
              depthWrite={!faded}
            />
          </mesh>
        )
      })}
      {hover && hovered && (
        <Html position={hover.point} style={{ pointerEvents: 'none' }}>
          <div className="hover-label">{hovered.name}</div>
        </Html>
      )}
    </group>
  )
}

/** Axial plane + point showing where the slice viewer's crosshair sits. */
function Crosshair({ crosshairRas, lo, hi }: { crosshairRas: Vec3 | null; lo: THREE.Vector3; hi: THREE.Vector3 }) {
  if (!crosshairRas) return null
  const p = rasToThree(crosshairRas)
  const w = Math.abs(hi.x - lo.x)
  const d = Math.abs(hi.z - lo.z)
  return (
    <group>
      <mesh position={[(lo.x + hi.x) / 2, p[1], (lo.z + hi.z) / 2]} rotation={[-Math.PI / 2, 0, 0]} renderOrder={2}>
        <planeGeometry args={[w, d]} />
        <meshBasicMaterial color="#38bdf8" transparent opacity={0.12} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <mesh position={p} renderOrder={3}>
        <sphereGeometry args={[4, 16, 16]} />
        <meshBasicMaterial color="#38bdf8" depthTest={false} />
      </mesh>
    </group>
  )
}

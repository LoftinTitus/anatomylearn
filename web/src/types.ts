export type Vec3 = [number, number, number]

export interface Structure {
  id: string
  /** Value of this structure in seg.nii.gz */
  label: number
  name: string
  system: string
  pathology: boolean
  color: Vec3
  snomed: { code: string; meaning: string; laterality: string | null } | null
  /** A point guaranteed to lie inside the structure, RAS mm */
  anchorRas: Vec3
  bboxRas: [Vec3, Vec3]
  volumeMl: number
  /** Structure touches the scan edge, so it is only partly captured */
  truncated: boolean
}

export interface Manifest {
  source: string
  segmentation: string
  rasToThree: [Vec3, Vec3, Vec3]
  volumeBoundsRas: [Vec3, Vec3]
  structures: Structure[]
}

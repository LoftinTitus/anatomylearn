import type { Vec3 } from './types'

// Must match RAS_TO_THREE in pipeline/build.py: three = (-R, S, A).
// The matrix is its own inverse's transpose (a rotation), so both directions are cheap.
export const rasToThree = ([x, y, z]: Vec3): Vec3 => [-x, z, y]
export const threeToRas = ([x, y, z]: Vec3): Vec3 => [-x, z, y]

import { Quaternion, Vector3 } from 'three'
import type { CameraFrame } from './types'

export function sampleCameraPose(frames: CameraFrame[], milliseconds: number) {
  if (!frames.length) return undefined
  let right = frames.findIndex((frame) => frame.t >= milliseconds)
  if (right < 0) right = frames.length - 1
  const a = frames[Math.max(0, right - 1)]
  const b = frames[right]
  const alpha = Math.max(0, Math.min(1, (milliseconds - a.t) / Math.max(1, b.t - a.t)))
  return {
    index: right,
    position: new Vector3(a.position.x, a.position.y, a.position.z).lerp(
      new Vector3(b.position.x, b.position.y, b.position.z), alpha,
    ),
    quaternion: new Quaternion(a.quaternion.x, a.quaternion.y, a.quaternion.z, a.quaternion.w)
      .normalize().slerp(
        new Quaternion(b.quaternion.x, b.quaternion.y, b.quaternion.z, b.quaternion.w).normalize(),
        alpha,
      ),
  }
}

/** Shared with the export renderer: centered, elevated 38-degree perspective. */
export function cameraMapGeometry(frames: CameraFrame[], width = 300, height = 150) {
  const positions = frames.map(({ position }) => new Vector3(position.x, position.y, position.z))
  const minimum = positions.reduce((a, b) => a.min(b), new Vector3(Infinity, Infinity, Infinity))
  const maximum = positions.reduce((a, b) => a.max(b), new Vector3(-Infinity, -Infinity, -Infinity))
  const center = minimum.clone().add(maximum).multiplyScalar(.5)
  const extent = maximum.clone().sub(minimum)
  const size = Math.max(extent.length(), 1)
  const eye = center.clone().add(new Vector3(0, size * .9, size * 1.45))
  const forward = center.clone().sub(eye).normalize()
  const right = forward.clone().cross(new Vector3(0, 1, 0)).normalize()
  const up = right.clone().cross(forward).normalize()
  const focal = .5 * height / Math.tan(38 * Math.PI / 360)
  const project = (point: Vector3) => {
    const relative = point.clone().sub(eye)
    const depth = Math.max(1e-6, relative.dot(forward))
    return [
      width / 2 + focal * relative.dot(right) / depth,
      height / 2 - focal * relative.dot(up) / depth,
    ] as const
  }
  const grid = []
  const half = size * 1.1
  const y = minimum.y - size * .08
  for (let i = 0; i <= 8; i++) {
    const offset = -half + size * 2.2 * i / 8
    grid.push([
      project(new Vector3(center.x - half, y, center.z + offset)),
      project(new Vector3(center.x + half, y, center.z + offset)),
    ], [
      project(new Vector3(center.x + offset, y, center.z - half)),
      project(new Vector3(center.x + offset, y, center.z + half)),
    ])
  }
  return { positions, project, grid, size }
}

export function cameraMarkerPoints(position: Vector3, quaternion: Quaternion, size: number) {
  return [
    [0, 0, 0], [-.065, -.045, -.13], [.065, -.045, -.13],
    [0, 0, 0], [.065, .045, -.13], [-.065, .045, -.13],
    [0, 0, 0], [-.065, -.045, -.13], [-.065, .045, -.13],
    [.065, .045, -.13], [.065, -.045, -.13],
  ].map(([x, y, z]) => new Vector3(x, y, z).multiplyScalar(size).applyQuaternion(quaternion).add(position))
}

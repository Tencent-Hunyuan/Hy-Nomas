import { Quaternion } from 'three'
import type { CameraFrame } from './types'

export const CAMERA_VISUAL_ROTATION_FACTOR = 0.75

export function visualRotationGain(userScale = 1) {
  return CAMERA_VISUAL_ROTATION_FACTOR * (
    Number.isFinite(userScale) && userScale > 0 ? userScale : 1
  )
}

/** Scale local angular increments, not quaternion components or wrapped Euler angles. */
export function scaledRotationStep(previous: Quaternion, current: Quaternion, gain: number) {
  const delta = previous.clone().invert().multiply(current).normalize()
  if (delta.w < 0) delta.set(-delta.x, -delta.y, -delta.z, -delta.w)
  const sinHalf = Math.hypot(delta.x, delta.y, delta.z)
  if (sinHalf < 1e-10) return new Quaternion()
  const halfAngle = Math.atan2(sinHalf, delta.w) * gain
  const factor = Math.sin(halfAngle) / sinHalf
  return new Quaternion(delta.x * factor, delta.y * factor, delta.z * factor, Math.cos(halfAngle))
}

/** Render-only frames: recorded positions, timestamps and source quaternions are untouched. */
export function visualCameraFrames(frames: CameraFrame[], userScale = 1): CameraFrame[] {
  const gain = visualRotationGain(userScale)
  let previous: Quaternion
  let orientation: Quaternion
  return frames.map((frame, index) => {
    const { x, y, z, w } = frame.quaternion
    const current = new Quaternion(x, y, z, w).normalize()
    if (index === 0) orientation = current.clone()
    else orientation.multiply(scaledRotationStep(previous, current, gain)).normalize()
    previous = current
    return {
      ...frame,
      quaternion: { x: orientation.x, y: orientation.y, z: orientation.z, w: orientation.w },
    }
  })
}

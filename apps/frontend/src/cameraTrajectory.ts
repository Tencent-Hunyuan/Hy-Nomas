import * as THREE from 'three'
import type { CameraControlKey, CameraFrame, CameraKeyEvent, CameraPath } from './types'

const STATIC_POSITION_EPSILON = 1e-5
const STATIC_ROTATION_EPSILON = 1e-4

function frameQuaternion(frame: CameraFrame) {
  return new THREE.Quaternion(
    frame.quaternion.x,
    frame.quaternion.y,
    frame.quaternion.z,
    frame.quaternion.w,
  ).normalize()
}

function cameraPoseIsStatic(previous: CameraFrame, current: CameraFrame) {
  const positionDelta = Math.hypot(
    current.position.x - previous.position.x,
    current.position.y - previous.position.y,
    current.position.z - previous.position.z,
  )
  const rotationDelta = frameQuaternion(previous).angleTo(frameQuaternion(current))
  return (
    positionDelta <= STATIC_POSITION_EPSILON
    && rotationDelta <= STATIC_ROTATION_EPSILON
  )
}

/**
 * Remove only the stationary suffix captured while the user waits before
 * submitting. One frame at the final pose is retained, and stationary pauses
 * in the middle of a trajectory are preserved if motion resumes afterward.
 */
export function trimTrailingStaticCameraFrames(
  frames: CameraFrame[],
): CameraFrame[] {
  let end = frames.length
  while (
    end > 1
    && cameraPoseIsStatic(frames[end - 2], frames[end - 1])
  ) {
    end -= 1
  }
  return frames.slice(0, end)
}

/** Keep key events within the retained trajectory and release held keys at its end. */
export function finishCameraKeyEvents(
  events: CameraKeyEvent[],
  durationMs: number,
): CameraKeyEvent[] {
  if (!Number.isFinite(durationMs) || durationMs <= 0) return events.map(event => ({ ...event }))
  const retained = events
    .filter(event => event.t <= durationMs)
    .map(event => ({ ...event }))
    .sort((a, b) => a.t - b.t)
  const held = new Set<CameraControlKey>()
  for (const event of retained) {
    if (event.pressed) held.add(event.code)
    else held.delete(event.code)
  }
  for (const code of held) retained.push({ t: durationMs, code, pressed: false })
  return retained
}

/** Repair saved/imported timelines before submission and keep overlays in sync. */
export function withFinishedCameraKeyEvents(path: CameraPath): CameraPath {
  const payload = JSON.parse(path.trajectory)
  if (!Array.isArray(payload.keyEvents) || typeof payload.durationMs !== 'number') return path
  const keyEvents = finishCameraKeyEvents(payload.keyEvents, payload.durationMs)
  return { ...path, keyEvents, trajectory: JSON.stringify({ ...payload, keyEvents }) }
}

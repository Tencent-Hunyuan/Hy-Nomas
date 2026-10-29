import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import type { CameraFrame, CameraPath } from './types'
import {
  finishCameraKeyEvents,
  trimTrailingStaticCameraFrames,
  withFinishedCameraKeyEvents,
} from './cameraTrajectory'

function frame(index: number, yawDegrees: number): CameraFrame {
  const q = new THREE.Quaternion().setFromEuler(
    new THREE.Euler(0, THREE.MathUtils.degToRad(yawDegrees), 0, 'YXZ'),
  )
  return {
    frame: index,
    t: Math.round(index * 1000 / 24),
    position: { x: 1, y: 2, z: 3 },
    quaternion: { x: q.x, y: q.y, z: q.z, w: q.w },
    fov: 55,
  }
}

describe('recorded camera trailing wait removal', () => {
  it('removes only duplicate poses at the end and retains the final pose', () => {
    const frames = [frame(0, 0), frame(1, 10), frame(2, 20), frame(3, 20), frame(4, 20)]
    expect(trimTrailingStaticCameraFrames(frames).map((item) => item.frame))
      .toEqual([0, 1, 2])
  })

  it('ignores FOV-only differences in the trailing stationary suffix', () => {
    const frames = [frame(0, 0), frame(1, 10), frame(2, 10)]
    frames[2].fov = 70
    expect(trimTrailingStaticCameraFrames(frames).map((item) => item.frame))
      .toEqual([0, 1])
  })

  it('preserves a stationary pause in the middle when movement resumes', () => {
    const frames = [
      frame(0, 0), frame(1, 10), frame(2, 10), frame(3, 10),
      frame(4, 20), frame(5, 20), frame(6, 20),
    ]
    expect(trimTrailingStaticCameraFrames(frames).map((item) => item.frame))
      .toEqual([0, 1, 2, 3, 4])
  })

  it('keeps one frame when the entire recording is stationary', () => {
    expect(trimTrailingStaticCameraFrames([frame(0, 0), frame(1, 0), frame(2, 0)]))
      .toHaveLength(1)
  })
})

describe('camera key timeline at recording end', () => {
  it.each([0.23, 13, 14.07])('closes a release %s ms beyond the trimmed trajectory', (excess) => {
    const events = [
      { t: 0, code: 'KeyW' as const, pressed: true },
      { t: 18125 + excess, code: 'KeyW' as const, pressed: false },
    ]
    const before = JSON.stringify(events)
    expect(finishCameraKeyEvents(events, 18125)).toEqual([
      events[0], { t: 18125, code: 'KeyW', pressed: false },
    ])
    expect(JSON.stringify(events)).toBe(before)
  })

  it('preserves middle pauses and releases held keys without introducing late presses', () => {
    expect(finishCameraKeyEvents([
      { t: 0, code: 'KeyW', pressed: true },
      { t: 50, code: 'KeyW', pressed: false },
      { t: 80, code: 'KeyW', pressed: true },
      { t: 90, code: 'ArrowLeft', pressed: true },
      { t: 110, code: 'KeyW', pressed: false },
      { t: 120, code: 'KeyD', pressed: true },
    ], 100)).toEqual([
      { t: 0, code: 'KeyW', pressed: true },
      { t: 50, code: 'KeyW', pressed: false },
      { t: 80, code: 'KeyW', pressed: true },
      { t: 90, code: 'ArrowLeft', pressed: true },
      { t: 100, code: 'KeyW', pressed: false },
      { t: 100, code: 'ArrowLeft', pressed: false },
    ])
  })

  it('keeps valid releases at the endpoint unchanged and is idempotent', () => {
    const events = [
      { t: 0, code: 'KeyW' as const, pressed: true },
      { t: 100, code: 'KeyW' as const, pressed: false },
    ]
    const finished = finishCameraKeyEvents(events, 100)
    expect(finished).toEqual(events)
    expect(finishCameraKeyEvents(finished, 100)).toEqual(finished)
    expect(finishCameraKeyEvents([], 100)).toEqual([])
  })

  it('repairs a saved path and its overlay without changing frames or duration', () => {
    const frames = [frame(0, 0), frame(1, 20)]
    const path: CameraPath = {
      id: 'old', name: 'Old', source: 'imported', frames,
      trajectory: JSON.stringify({ frames, durationMs: 100, keyEvents: [
        { t: 0, code: 'KeyW', pressed: true }, { t: 113, code: 'KeyW', pressed: false },
      ] }),
    }
    const fixed = withFinishedCameraKeyEvents(path)
    const payload = JSON.parse(fixed.trajectory)
    expect(payload.keyEvents).toEqual(fixed.keyEvents)
    expect(payload.keyEvents.at(-1)).toEqual({ t: 100, code: 'KeyW', pressed: false })
    expect(payload.frames).toEqual(frames)
    expect(payload.durationMs).toBe(100)
    expect(JSON.parse(path.trajectory).keyEvents.at(-1).t).toBe(113)
  })
})

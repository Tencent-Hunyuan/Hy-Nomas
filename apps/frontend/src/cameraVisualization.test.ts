import { describe, expect, it } from 'vitest'
import { Quaternion, Vector3 } from 'three'
import { scaledRotationStep, visualCameraFrames, visualRotationGain } from './cameraVisualization'
import { sampleCameraPose } from './cameraMap'
import type { CameraFrame } from './types'

function frame(degrees: number, index: number): CameraFrame {
  const q = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), degrees * Math.PI / 180)
  return { frame: index, t: index * 1000, position: { x: index, y: 0, z: 0 },
    quaternion: { x: q.x, y: q.y, z: q.z, w: q.w }, fov: 55 }
}

describe('render-only camera rotation gain', () => {
  it('uses 0.75 times the user setting, including gains above one', () => {
    expect(visualRotationGain(2)).toBe(1.5)
    const raw = [frame(0, 0), frame(40, 1)]
    const saved = JSON.stringify(raw)
    const result = visualCameraFrames(raw, 2)
    const pose = sampleCameraPose(result, 1000)!
    expect(pose.quaternion.angleTo(new Quaternion()) * 180 / Math.PI).toBeCloseTo(60)
    expect(JSON.stringify(raw)).toBe(saved)
    expect(result[1].t).toBe(raw[1].t)
    expect(result[1].position).toEqual(raw[1].position)
  })

  it('matches the recorder incremental preview and supports wraparound and seeking', () => {
    const raw = [frame(0, 0), frame(120, 1), frame(240, 2), frame(360, 3)]
    const result = visualCameraFrames(raw, 1)
    const live = new Quaternion()
    for (let i = 1; i < raw.length; i++) {
      const a = raw[i - 1].quaternion
      const b = raw[i].quaternion
      live.multiply(scaledRotationStep(
        new Quaternion(a.x, a.y, a.z, a.w),
        new Quaternion(b.x, b.y, b.z, b.w), .75,
      ))
    }
    expect(sampleCameraPose(result, 3000)!.quaternion.angleTo(live)).toBeCloseTo(0)
    expect(sampleCameraPose(result, 500)!.quaternion.angleTo(new Quaternion()) * 180 / Math.PI).toBeCloseTo(45)
    expect(sampleCameraPose(result, 0)!.quaternion.angleTo(new Quaternion())).toBeCloseTo(0)
  })
})

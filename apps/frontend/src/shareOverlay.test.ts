import { describe, expect, it } from 'vitest'
import type { PromptTurn } from './components/PromptTurnTimeline'
import { buildSharedVideoOverlay } from './shareOverlay'

describe('buildSharedVideoOverlay', () => {
  it('serializes timed prompt and camera turns for the public HTML player', () => {
    const turns: PromptTurn[] = [
      {
        id: 1,
        number: 1,
        prompt: 'Walk toward the bridge',
        submittedAt: '08:00:00',
        stage: 'complete',
        startSeconds: 0,
        endSeconds: 5,
      },
      {
        id: 2,
        number: 2,
        prompt: 'Generate from the selected camera path.',
        submittedAt: '08:01:00',
        stage: 'complete',
        startSeconds: 6,
        endSeconds: 9,
        cameraPath: {
          id: 'camera-1',
          name: 'Custom path 08:01',
          source: 'recorded',
          trajectory: '{}',
          durationMs: 3000,
          sampleRate: 24,
          visualizationRotationScale: 3.2,
          frames: [
            {
              frame: 0,
              t: 0,
              position: { x: 0, y: 0, z: 0 },
              quaternion: { x: 0, y: 0, z: 0, w: 1 },
              fov: 55,
            },
            {
              frame: 1,
              t: 1000,
              position: { x: 1, y: 0, z: -1 },
              quaternion: { x: 0, y: 0, z: 0, w: 1 },
              fov: 55,
            },
          ],
          keyEvents: [
            { t: 10, code: 'KeyW', pressed: true },
          ],
        },
      },
      {
        id: 3,
        number: 3,
        prompt: 'Not generated yet',
        submittedAt: '08:02:00',
        stage: 'rendering',
      },
    ]

    expect(buildSharedVideoOverlay(turns)).toEqual({
      version: 1,
      turns: [
        {
          number: 1,
          prompt: 'Walk toward the bridge',
          start_seconds: 0,
          end_seconds: 5,
        },
        {
          number: 2,
          prompt: 'Generate from the selected camera path.',
          start_seconds: 6,
          end_seconds: 9,
          camera: {
            name: 'Custom path 08:01',
            duration_ms: 3000,
            sample_rate: 24,
            visualization_rotation_scale: 3.2,
            frames: turns[1].cameraPath?.frames,
            key_events: turns[1].cameraPath?.keyEvents,
          },
        },
      ],
    })
  })
})

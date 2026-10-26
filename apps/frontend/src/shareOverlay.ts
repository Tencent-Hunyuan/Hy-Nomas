import type { PromptTurn } from './components/PromptTurnTimeline'
import type { CameraFrame, CameraKeyEvent, CameraPath } from './types'

export interface SharedCameraOverlay {
  name: string
  duration_ms?: number
  sample_rate?: number
  visualization_rotation_scale?: number
  frames: CameraFrame[]
  key_events: CameraKeyEvent[]
}

export interface SharedOverlayTurn {
  number: number
  prompt: string
  start_seconds: number
  end_seconds: number
  camera?: SharedCameraOverlay
}

export interface SharedVideoOverlay {
  version: 1
  theme?: 'dark' | 'light'
  turns: SharedOverlayTurn[]
}

function parsedCameraPayload(path: CameraPath) {
  try {
    const value = JSON.parse(path.trajectory) as {
      frames?: CameraFrame[]
      keyEvents?: CameraKeyEvent[]
      durationMs?: number
      sampleRate?: number
      name?: string
    }
    return value && typeof value === 'object' ? value : undefined
  } catch {
    return undefined
  }
}

function serializeCameraPath(path: CameraPath): SharedCameraOverlay | undefined {
  const parsed = parsedCameraPayload(path)
  const frames = path.frames ?? parsed?.frames ?? []
  if (!frames.length) return undefined
  return {
    name: path.name || parsed?.name || 'Camera path',
    duration_ms: path.durationMs ?? parsed?.durationMs,
    sample_rate: path.sampleRate ?? parsed?.sampleRate,
    ...(path.visualizationRotationScale !== undefined
      ? { visualization_rotation_scale: path.visualizationRotationScale } : {}),
    frames,
    key_events: path.keyEvents ?? parsed?.keyEvents ?? [],
  }
}

export function buildSharedVideoOverlay(
  turns: PromptTurn[],
  theme?: 'dark' | 'light',
): SharedVideoOverlay {
  return {
    version: 1,
    ...(theme ? { theme } : {}),
    turns: turns.flatMap((turn) => {
      if (
        !Number.isFinite(turn.startSeconds)
        || !Number.isFinite(turn.endSeconds)
        || (turn.endSeconds ?? 0) <= (turn.startSeconds ?? 0)
      ) {
        return []
      }
      const camera = turn.cameraPath
        ? serializeCameraPath(turn.cameraPath)
        : undefined
      return [{
        number: turn.number,
        prompt: turn.prompt,
        start_seconds: Number(turn.startSeconds),
        end_seconds: Number(turn.endSeconds),
        ...(camera ? { camera } : {}),
      }]
    }),
  }
}

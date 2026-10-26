import { useEffect, useState, type RefObject } from 'react'
import type { CameraFrame, CameraKeyEvent, CameraPath } from '../types'
import { CameraPathMap } from './CameraPathMap'
import { OverlayVisibilityToggle } from './OverlayVisibilityToggle'

interface CameraPlaybackOverlayProps {
  path: CameraPath
  videoRef: RefObject<HTMLVideoElement | null>
  segmentStartSeconds: number
  segmentDurationSeconds: number
  visible: boolean
  onVisibilityToggle: () => void
}

const IDLE_ACTIONS = {
  forward: false,
  backward: false,
  left: false,
  right: false,
  lookUp: false,
  lookDown: false,
  lookLeft: false,
  lookRight: false,
}

const EMPTY_FRAMES: CameraFrame[] = []
const ACTION_BY_CODE = {
  KeyW: 'forward', KeyA: 'left', KeyS: 'backward', KeyD: 'right',
  ArrowUp: 'lookUp', ArrowDown: 'lookDown',
  ArrowLeft: 'lookLeft', ArrowRight: 'lookRight',
} as const
const KEY_GROUPS = [
  {
    label: 'MOVE',
    aria: 'Camera movement',
    keys: [['W', 'forward'], ['A', 'left'], ['S', 'backward'], ['D', 'right']],
  },
  {
    label: 'LOOK',
    aria: 'Camera rotation',
    keys: [['↑', 'lookUp'], ['←', 'lookLeft'], ['↓', 'lookDown'], ['→', 'lookRight']],
  },
] as const

export function actionsAtTime(events: CameraKeyEvent[], milliseconds: number) {
  const actions = { ...IDLE_ACTIONS }
  for (const event of events) {
    if (event.t > milliseconds) break
    actions[ACTION_BY_CODE[event.code]] = event.pressed
  }
  return actions
}

export function cameraSegmentIsActive(localSeconds: number, duration: number) {
  return localSeconds >= 0 && localSeconds < Math.max(0, duration)
}

export function CameraPlaybackOverlay({
  path, videoRef, segmentStartSeconds, segmentDurationSeconds,
  visible, onVisibilityToggle,
}: CameraPlaybackOverlayProps) {
  const [actions, setActions] = useState(IDLE_ACTIONS)
  const [progress, setProgress] = useState(0)
  const frames = path.frames ?? EMPTY_FRAMES
  const durationMs = path.durationMs
    ?? frames.at(-1)?.t
    ?? Math.max(0, frames.length - 1) * 1000 / (path.sampleRate ?? 24)

  // Key telemetry must work without WebGL and while seeking or paused.
  useEffect(() => {
    if (!visible) return
    let request = 0
    let previousIndex = -2
    const events = path.keyEvents ?? []
    const update = () => {
      const video = videoRef.current
      const local = video ? video.currentTime - segmentStartSeconds : -1
      const active = cameraSegmentIsActive(local, segmentDurationSeconds)
      const progress = Math.max(0, Math.min(1, local / Math.max(0.001, segmentDurationSeconds)))
      setProgress(progress)
      const time = active ? progress * durationMs : -1
      const index = active ? events.findLastIndex((event) => event.t <= time) : -1
      if (index !== previousIndex) {
        previousIndex = index
        setActions(active ? actionsAtTime(events, time) : IDLE_ACTIONS)
      }
      request = requestAnimationFrame(update)
    }
    request = requestAnimationFrame(update)
    return () => cancelAnimationFrame(request)
  }, [path.keyEvents, durationMs, videoRef, segmentStartSeconds, segmentDurationSeconds, visible])

  return (
    <>
      <OverlayVisibilityToggle
        kind="camera"
        visible={visible}
        onToggle={onVisibilityToggle}
      />
      {visible && (
        <>
          <CameraPathMap
            frames={frames}
            durationMs={durationMs}
            progress={progress}
            name={path.name}
            rotationScale={path.visualizationRotationScale}
          />
          <div className="demo-camera-overlay" aria-label="Camera action overlay">
            {KEY_GROUPS.map((group) => (
              <div className="demo-key-group" aria-label={group.aria} key={group.label}>
                <div className="demo-key-grid">
                  {group.keys.map(([label, action]) => (
                    <kbd key={label} className={actions[action] ? 'active' : ''}>{label}</kbd>
                  ))}
                </div>
                <span>{group.label}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </>
  )
}

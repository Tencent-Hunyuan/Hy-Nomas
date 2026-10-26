import { useMemo } from 'react'
import { cameraMapGeometry, cameraMarkerPoints, sampleCameraPose } from '../cameraMap'
import { visualCameraFrames } from '../cameraVisualization'
import type { CameraFrame } from '../types'

interface CameraPathMapProps {
  frames: CameraFrame[]
  durationMs: number
  progress: number
  name: string
  rotationScale?: number
}

const pointsText = (points: ReadonlyArray<readonly [number, number]>) => (
  points.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ')
)

export function CameraPathMap({
  frames: rawFrames, durationMs, progress, name, rotationScale = 1,
}: CameraPathMapProps) {
  const frames = useMemo(
    () => visualCameraFrames(rawFrames, rotationScale),
    [rawFrames, rotationScale],
  )
  const geometry = useMemo(
    () => rawFrames.length ? cameraMapGeometry(rawFrames) : undefined,
    [rawFrames],
  )
  if (!geometry) return null
  const clamped = Math.max(0, Math.min(1, progress))
  const sample = sampleCameraPose(frames, frames[0].t + durationMs * clamped)!
  const { project, positions, grid, size } = geometry
  const travelled = positions.slice(0, sample.index).concat(sample.position)
  const marker = cameraMarkerPoints(sample.position, sample.quaternion, size)
  const current = project(sample.position)
  return (
    <aside className="demo-camera-map" aria-label="Camera path visualization">
      <header>CAMERA PATH</header>
      <svg viewBox="0 0 300 150" aria-label="Camera trajectory and orientation" role="img">
        <g className="demo-map-grid">
          {grid.map(([a, b], index) => (
            <line key={index} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} />
          ))}
        </g>
        <polyline className="demo-map-path" points={pointsText(positions.map(project))} />
        <polyline className="demo-map-travelled" points={pointsText(travelled.map(project))} />
        {[positions[0], positions.at(-1)!].map((point, index) => {
          const [x, y] = project(point)
          return <circle key={index} className="demo-map-endpoint" cx={x} cy={y} r="2.5" />
        })}
        <polyline className="demo-map-marker" points={pointsText(marker.map(project))} />
        <circle className="demo-map-endpoint" cx={current[0]} cy={current[1]} r="2.5" />
      </svg>
      <footer>
        <span>{name}</span>
        <strong>{Math.round(clamped * 100)}%</strong>
      </footer>
    </aside>
  )
}

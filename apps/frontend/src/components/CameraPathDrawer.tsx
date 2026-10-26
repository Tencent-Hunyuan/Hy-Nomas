import {
  ArrowLeft,
  Download,
  RotateCcw,
  Upload,
} from 'lucide-react'
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  type ChangeEvent,
} from 'react'
import * as THREE from 'three'
import { scaledRotationStep, visualRotationGain } from '../cameraVisualization'
import {
  trimTrailingStaticCameraFrames,
} from '../cameraTrajectory'
import type {
  CameraControlKey,
  CameraFrame,
  CameraKeyEvent,
  CameraPath,
} from '../types'

// Camera conditioning is trained and served on the model's 24 FPS timeline.
// Record at the same rate so one browser sample corresponds to one model
// pixel-frame sample before the first-frame-aware VAE alignment.
const SAMPLE_RATE = 24
const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE
const ROTATION_SENSITIVITY_SCALE = 0.6
const KEYBOARD_TURN_SPEED = THREE.MathUtils.degToRad(20)
const POINTER_TURN_SENSITIVITY = 0.004 * ROTATION_SENSITIVITY_SCALE
const INITIAL_CAMERA_POSITION = [0, 1.5, 10] as const
const INITIAL_CAMERA_TARGET = [0, 1.5, 0] as const
const RECORDED_CONTROL_KEYS = [
  'KeyW',
  'KeyA',
  'KeyS',
  'KeyD',
  'ArrowUp',
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
] as const

function isRecordedControlKey(code: string): code is CameraControlKey {
  return RECORDED_CONTROL_KEYS.includes(code as CameraControlKey)
}

function normalizeImportedKeyEvents(value: unknown): CameraKeyEvent[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    if (!item || typeof item !== 'object') return []
    const raw = item as Partial<CameraKeyEvent>
    const t = Number(raw.t)
    const code = String(raw.code ?? '')
    if (!Number.isFinite(t) || t < 0 || !isRecordedControlKey(code)) return []
    return [{ t, code, pressed: raw.pressed === true }]
  }).sort((left, right) => left.t - right.t)
}

interface CameraPathDrawerProps {
  selected?: CameraPath
  disabled?: boolean
  embedded?: boolean
  theme: 'dark' | 'light'
  translationScale?: number
  rotationScale?: number
  onTranslationScaleChange: (value: number) => void
  onRotationScaleChange: (value: number) => void
  onBack?: () => void
  onApply: (path: CameraPath) => void
  onClear: () => void
  onRecordingStateChange?: (recording: boolean) => void
}

export interface CameraPathDrawerHandle {
  finalizeRecording: () => CameraPath | undefined
  resetRecording: () => void
}

function makeId() {
  return `camera-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

function formatDuration(milliseconds = 0) {
  const seconds = milliseconds / 1000
  return `${seconds.toFixed(2)}s`
}

function calculateDistance(frames: CameraFrame[]) {
  return frames.slice(1).reduce((distance, frame, index) => {
    const previous = frames[index].position
    return distance + Math.hypot(
      frame.position.x - previous.x,
      frame.position.y - previous.y,
      frame.position.z - previous.z,
    )
  }, 0)
}

function normalizeSampleRate(value: unknown) {
  const sampleRate = Number(value)
  return Number.isFinite(sampleRate) && sampleRate > 0 ? sampleRate : SAMPLE_RATE
}

function inferredTrajectoryDurationMs(frames: CameraFrame[], sampleRate: number) {
  if (!frames.length) return 0
  const intervalMs = 1000 / normalizeSampleRate(sampleRate)
  const firstTimestamp = Number(frames[0]?.t)
  const lastTimestamp = Number(frames.at(-1)?.t)
  const timestampDuration =
    Number.isFinite(firstTimestamp)
    && Number.isFinite(lastTimestamp)
    && lastTimestamp >= firstTimestamp
      ? lastTimestamp - firstTimestamp + intervalMs
      : 0
  return Math.max(
    intervalMs,
    frames.length * intervalMs,
    timestampDuration,
  )
}

function createTrajectoryPayload(
  name: string,
  frames: CameraFrame[],
  keyEvents: CameraKeyEvent[],
  totalDistance: number,
  durationMs: number,
) {
  return {
    version: '1.0',
    type: 'camera-trajectory',
    name,
    recordedAt: new Date().toISOString(),
    sampleRate: SAMPLE_RATE,
    durationMs,
    totalDistance: Number(totalDistance.toFixed(6)),
    coordinateSystem: {
      handedness: 'right',
      up: '+Y',
      forward: '-Z',
      unit: 'meter',
    },
    frames,
    keyEvents,
  }
}

function normalizeImportedFrames(value: unknown, sampleRate: number): CameraFrame[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item, index) => {
    if (!item || typeof item !== 'object') return []
    const raw = item as Partial<CameraFrame>
    const position = raw.position
    const quaternion = raw.quaternion
    if (
      !position ||
      !quaternion ||
      ![position.x, position.y, position.z, quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        .every((number) => Number.isFinite(Number(number)))
    ) {
      return []
    }
    return [{
      frame: Number.isFinite(Number(raw.frame)) ? Number(raw.frame) : index,
      t: Number.isFinite(Number(raw.t))
        ? Number(raw.t)
        : Math.round(index * 1000 / sampleRate),
      position: {
        x: Number(position.x),
        y: Number(position.y),
        z: Number(position.z),
      },
      quaternion: {
        x: Number(quaternion.x),
        y: Number(quaternion.y),
        z: Number(quaternion.z),
        w: Number(quaternion.w),
      },
      fov: Number.isFinite(Number(raw.fov)) ? Number(raw.fov) : 55,
    }]
  })
}

interface CameraRecorderHandle {
  finalizeRecording: () => CameraPath | undefined
  resetRecording: () => void
}

interface CameraRecorderProps {
  theme: 'dark' | 'light'
  disabled?: boolean
  translationScale?: number
  rotationScale?: number
  onTranslationScaleChange: (value: number) => void
  onRotationScaleChange: (value: number) => void
  onRecorded: (path: CameraPath) => void
  onRecordingStateChange?: (recording: boolean) => void
}

const CameraRecorder = forwardRef<CameraRecorderHandle, CameraRecorderProps>(function CameraRecorder(
  {
    theme, disabled, onRecorded, onRecordingStateChange,
    translationScale, rotationScale, onTranslationScaleChange, onRotationScaleChange,
  },
  ref,
) {
  const mountRef = useRef<HTMLDivElement>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const visualScaleRef = useRef(rotationScale)
  const resetViewRef = useRef<(() => void) | null>(null)
  const onRecordedRef = useRef(onRecorded)
  const onRecordingStateChangeRef = useRef(onRecordingStateChange)
  const recordingRef = useRef(false)
  const framesRef = useRef<CameraFrame[]>([])
  const pressedKeysRef = useRef<Set<string>>(new Set())
  const keyEventsRef = useRef<CameraKeyEvent[]>([])
  const startTimeRef = useRef(0)
  const actualDurationMsRef = useRef(0)
  const distanceRef = useRef(0)
  const [recording, setRecording] = useState(false)
  const [durationMs, setDurationMs] = useState(0)
  const [frameCount, setFrameCount] = useState(0)
  const [distance, setDistance] = useState(0)
  const [rotationScaleInput, setRotationScaleInput] = useState(
    rotationScale === undefined ? '' : String(rotationScale),
  )
  const [translationScaleInput, setTranslationScaleInput] = useState(
    translationScale === undefined ? '' : String(translationScale),
  )
  useEffect(() => {
    visualScaleRef.current = rotationScale
    setRotationScaleInput(rotationScale === undefined ? '' : String(rotationScale))
  }, [rotationScale])
  useEffect(() => {
    setTranslationScaleInput(translationScale === undefined ? '' : String(translationScale))
  }, [translationScale])
  const [viewerAvailable, setViewerAvailable] = useState(true)
  const [activeKeys, setActiveKeys] = useState<Set<string>>(() => new Set())

  onRecordedRef.current = onRecorded
  onRecordingStateChangeRef.current = onRecordingStateChange

  const startRecording = () => {
    if (recordingRef.current) return
    framesRef.current = []
    keyEventsRef.current = []
    distanceRef.current = 0
    startTimeRef.current = performance.now()
    actualDurationMsRef.current = 0
    recordingRef.current = true
    setDurationMs(0)
    setFrameCount(0)
    setDistance(0)
    setRecording(true)
    onRecordingStateChangeRef.current?.(true)
  }

  const captureThrough = useCallback((timestamp: number, camera = cameraRef.current) => {
    if (!recordingRef.current || !camera) return false
    const elapsedMs = Math.max(0, timestamp - startTimeRef.current)
    actualDurationMsRef.current = elapsedMs
    const targetFrameCount = Math.max(
      1,
      Math.round(elapsedMs / SAMPLE_INTERVAL_MS),
    )
    const previousFrameCount = framesRef.current.length
    while (framesRef.current.length < targetFrameCount) {
      const previousFrame = framesRef.current.at(-1)
      const position = camera.position
      if (previousFrame) {
        distanceRef.current += Math.hypot(
          position.x - previousFrame.position.x,
          position.y - previousFrame.position.y,
          position.z - previousFrame.position.z,
        )
      }
      framesRef.current.push({
        frame: framesRef.current.length,
        t: Math.round(framesRef.current.length * SAMPLE_INTERVAL_MS),
        position: { x: position.x, y: position.y, z: position.z },
        quaternion: {
          x: camera.quaternion.x,
          y: camera.quaternion.y,
          z: camera.quaternion.z,
          w: camera.quaternion.w,
        },
        fov: camera.fov,
      })
    }
    const addedFrames = framesRef.current.length > previousFrameCount
    if (addedFrames) {
      setDurationMs(elapsedMs)
      setFrameCount(framesRef.current.length)
      setDistance(distanceRef.current)
    }
    return addedFrames
  }, [])

  const finishRecording = () => {
    if (!recordingRef.current) return undefined
    captureThrough(performance.now())
    if (!framesRef.current.length) return undefined
    recordingRef.current = false
    setRecording(false)
    onRecordingStateChangeRef.current?.(false)
    const frames = trimTrailingStaticCameraFrames(
      framesRef.current.map((frame) => ({
        ...frame,
        position: { ...frame.position },
        quaternion: { ...frame.quaternion },
      })),
    )
    const durationMs = Math.max(
      frames.length * SAMPLE_INTERVAL_MS,
      SAMPLE_INTERVAL_MS,
    )
    const totalDistance = calculateDistance(frames)
    framesRef.current = frames
    actualDurationMsRef.current = durationMs
    distanceRef.current = totalDistance
    setDurationMs(durationMs)
    setFrameCount(frames.length)
    setDistance(totalDistance)
    const name = `Custom path ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
    const payload = createTrajectoryPayload(
      name,
      frames,
      keyEventsRef.current,
      totalDistance,
      durationMs,
    )
    const path: CameraPath = {
      id: makeId(),
      name,
      source: 'recorded',
      trajectory: JSON.stringify(payload),
      durationMs: payload.durationMs,
      sampleRate: SAMPLE_RATE,
      totalDistance,
      frames,
      keyEvents: keyEventsRef.current.map((event) => ({ ...event })),
    }
    onRecordedRef.current(path)
    return path
  }

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true })
    } catch {
      setViewerAvailable(false)
      return
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    const lightTheme = theme === 'light'
    const palette = lightTheme
      ? {
          background: 0xe8e1d5,
          fog: 0xe8e1d5,
          hemisphereSky: 0xfffbf3,
          hemisphereGround: 0xa59b88,
          gridMajor: 0x607b45,
          gridMinor: 0xc6bdad,
          backGridMajor: 0x8c9a79,
          backGridMinor: 0xd8d0c2,
          subject: 0x607b45,
          path: 0x526e39,
        }
      : {
          background: 0x171a20,
          fog: 0x171a20,
          hemisphereSky: 0xaabbd4,
          hemisphereGround: 0x181a1f,
          gridMajor: 0xc8f259,
          gridMinor: 0x3b424d,
          backGridMajor: 0x667080,
          backGridMinor: 0x303640,
          subject: 0xc8f259,
          path: 0xc8f259,
        }
    renderer.setClearColor(palette.background, 1)
    renderer.domElement.tabIndex = 0
    mount.append(renderer.domElement)

    const scene = new THREE.Scene()
    scene.fog = new THREE.FogExp2(palette.fog, 0.022)
    const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 1000)
    camera.position.set(...INITIAL_CAMERA_POSITION)
    camera.lookAt(...INITIAL_CAMERA_TARGET)
    cameraRef.current = camera

    scene.add(
      new THREE.HemisphereLight(
        palette.hemisphereSky,
        palette.hemisphereGround,
        lightTheme ? 2.15 : 1.7,
      ),
    )
    const light = new THREE.DirectionalLight(0xffffff, 2)
    light.position.set(7, 10, 5)
    scene.add(light)

    const grid = new THREE.GridHelper(
      80,
      80,
      palette.gridMajor,
      palette.gridMinor,
    )
    const gridMaterials = Array.isArray(grid.material) ? grid.material : [grid.material]
    gridMaterials.forEach((material) => {
      material.transparent = true
      material.opacity = 0.58
    })
    scene.add(grid)

    // A subtle vertical grid makes pitch changes readable even when the camera
    // is not translating across the ground plane.
    const backGrid = new THREE.GridHelper(
      42,
      28,
      palette.backGridMajor,
      palette.backGridMinor,
    )
    backGrid.rotation.x = Math.PI / 2
    backGrid.position.set(0, 10, -15)
    const backGridMaterials = Array.isArray(backGrid.material)
      ? backGrid.material
      : [backGrid.material]
    backGridMaterials.forEach((material) => {
      material.transparent = true
      material.opacity = 0.34
      material.depthWrite = false
    })
    scene.add(backGrid)

    const subject = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.15, 1),
      new THREE.MeshStandardMaterial({
        color: palette.subject,
        emissive: palette.subject,
        emissiveIntensity: lightTheme ? 0.06 : 0.16,
        roughness: 0.38,
        metalness: 0.28,
        flatShading: true,
      }),
    )
    subject.position.y = 1.5
    scene.add(subject)

    const pathGeometry = new THREE.BufferGeometry()
    const path = new THREE.Line(
      pathGeometry,
      new THREE.LineBasicMaterial({ color: palette.path }),
    )
    scene.add(path)

    const keys = pressedKeysRef.current
    let rightDragging = false
    let lastX = 0
    let lastY = 0
    const rotation = new THREE.Euler(0, 0, 0, 'YXZ')
    rotation.setFromQuaternion(camera.quaternion)
    // Keep the recording camera raw. Only this separate render camera gets
    // the user's rotation gain * 0.75; model conditioning is unchanged.
    const visualCamera = camera.clone()
    let visualGain = visualRotationGain(visualScaleRef.current)
    let rawOrientations = [camera.quaternion.clone()]
    const visualOrientation = camera.quaternion.clone()
    const updateVisualOrientation = () => {
      const current = camera.quaternion
      const previous = rawOrientations.at(-1)!
      const gain = visualRotationGain(visualScaleRef.current)
      if (gain !== visualGain) {
        visualGain = gain
        visualOrientation.copy(rawOrientations[0])
        for (let i = 1; i < rawOrientations.length; i++) {
          visualOrientation.multiply(scaledRotationStep(rawOrientations[i - 1], rawOrientations[i], gain))
        }
      }
      if (previous.angleTo(current) > 1e-10) {
        visualOrientation.multiply(scaledRotationStep(previous, current, gain)).normalize()
        rawOrientations.push(current.clone())
      }
    }
    resetViewRef.current = () => {
      keys.clear()
      rightDragging = false
      setActiveKeys(new Set())
      camera.position.set(...INITIAL_CAMERA_POSITION)
      camera.lookAt(...INITIAL_CAMERA_TARGET)
      rotation.setFromQuaternion(camera.quaternion)
      rawOrientations = [camera.quaternion.clone()]
      visualOrientation.copy(camera.quaternion)
      pathGeometry.setAttribute(
        'position',
        new THREE.Float32BufferAttribute([], 3),
      )
    }

    const resize = () => {
      const { clientWidth, clientHeight } = mount
      if (!clientWidth || !clientHeight) return
      renderer.setSize(clientWidth, clientHeight, false)
      camera.aspect = clientWidth / clientHeight
      camera.updateProjectionMatrix()
    }

    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(resize)
    observer?.observe(mount)
    resize()

    const onKeyDown = (event: KeyboardEvent) => {
      if (['INPUT', 'TEXTAREA'].includes((event.target as HTMLElement).tagName)) return
      const isControlKey = isRecordedControlKey(event.code)
      if (!isControlKey && !event.code.startsWith('Shift')) return
      if (isControlKey) {
        if (!recordingRef.current) startRecording()
        if (!keys.has(event.code)) {
          keyEventsRef.current.push({
            t: Math.max(0, performance.now() - startTimeRef.current),
            code: event.code,
            pressed: true,
          })
        }
      }
      keys.add(event.code)
      setActiveKeys(new Set(keys))
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(event.code)) {
        event.preventDefault()
      }
    }
    const onKeyUp = (event: KeyboardEvent) => {
      if (keys.has(event.code) && isRecordedControlKey(event.code)) {
        keyEventsRef.current.push({
          t: Math.max(0, performance.now() - startTimeRef.current),
          code: event.code,
          pressed: false,
        })
      }
      keys.delete(event.code)
      setActiveKeys(new Set(keys))
    }
    const onWindowBlur = () => {
      const t = Math.max(0, performance.now() - startTimeRef.current)
      keys.forEach((code) => {
        if (isRecordedControlKey(code)) {
          keyEventsRef.current.push({ t, code, pressed: false })
        }
      })
      keys.clear()
      setActiveKeys(new Set())
    }
    const onPointerDown = (event: PointerEvent) => {
      renderer.domElement.focus()
      if (event.button !== 2) return
      rightDragging = true
      lastX = event.clientX
      lastY = event.clientY
      renderer.domElement.setPointerCapture(event.pointerId)
    }
    const onPointerMove = (event: PointerEvent) => {
      if (!rightDragging) return
      if (event.clientX !== lastX || event.clientY !== lastY) startRecording()
      rotation.y -= (event.clientX - lastX) * POINTER_TURN_SENSITIVITY
      rotation.x -= (event.clientY - lastY) * POINTER_TURN_SENSITIVITY
      camera.quaternion.setFromEuler(rotation)
      updateVisualOrientation()
      lastX = event.clientX
      lastY = event.clientY
    }
    const onPointerUp = () => {
      rightDragging = false
    }
    const onContextMenu = (event: MouseEvent) => event.preventDefault()

    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    window.addEventListener('blur', onWindowBlur)
    renderer.domElement.addEventListener('pointerdown', onPointerDown)
    renderer.domElement.addEventListener('pointermove', onPointerMove)
    renderer.domElement.addEventListener('pointerup', onPointerUp)
    renderer.domElement.addEventListener('contextmenu', onContextMenu)

    let animationFrame = 0
    let previous = performance.now()
    const animate = (timestamp: number) => {
      const delta = Math.min((timestamp - previous) / 1000, 0.05)
      previous = timestamp
      const speed = keys.has('ShiftLeft') || keys.has('ShiftRight') ? 7 : 2.8
      const direction = new THREE.Vector3()
      camera.getWorldDirection(direction)
      const right = new THREE.Vector3().crossVectors(direction, camera.up).normalize()

      if (keys.has('KeyW')) camera.position.addScaledVector(direction, speed * delta)
      if (keys.has('KeyS')) camera.position.addScaledVector(direction, -speed * delta)
      if (keys.has('KeyA')) camera.position.addScaledVector(right, -speed * delta)
      if (keys.has('KeyD')) camera.position.addScaledVector(right, speed * delta)

      const turnSpeed = KEYBOARD_TURN_SPEED * delta
      if (keys.has('ArrowLeft')) rotation.y += turnSpeed
      if (keys.has('ArrowRight')) rotation.y -= turnSpeed
      if (keys.has('ArrowUp')) rotation.x += turnSpeed
      if (keys.has('ArrowDown')) rotation.x -= turnSpeed
      if ([...keys].some((key) => key.startsWith('Arrow'))) {
        camera.quaternion.setFromEuler(rotation)
      }

      if (recordingRef.current && captureThrough(timestamp, camera)) {
        const positions = framesRef.current.flatMap((frame) => [
          frame.position.x,
          frame.position.y,
          frame.position.z,
        ])
        pathGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
      }

      subject.rotation.y += delta * 0.18
      updateVisualOrientation()
      visualCamera.copy(camera)
      visualCamera.quaternion.copy(visualOrientation)
      renderer.render(scene, visualCamera)
      animationFrame = requestAnimationFrame(animate)
    }
    animationFrame = requestAnimationFrame(animate)

    return () => {
      cancelAnimationFrame(animationFrame)
      observer?.disconnect()
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
      window.removeEventListener('blur', onWindowBlur)
      renderer.domElement.removeEventListener('pointerdown', onPointerDown)
      renderer.domElement.removeEventListener('pointermove', onPointerMove)
      renderer.domElement.removeEventListener('pointerup', onPointerUp)
      renderer.domElement.removeEventListener('contextmenu', onContextMenu)
      renderer.dispose()
      pathGeometry.dispose()
      grid.geometry.dispose()
      gridMaterials.forEach((material) => material.dispose())
      backGrid.geometry.dispose()
      backGridMaterials.forEach((material) => material.dispose())
      mount.removeChild(renderer.domElement)
      resetViewRef.current = null
    }
  }, [captureThrough, theme])

  const resetCamera = () => {
    recordingRef.current = false
    framesRef.current = []
    keyEventsRef.current = []
    actualDurationMsRef.current = 0
    distanceRef.current = 0
    setRecording(false)
    setDurationMs(0)
    setFrameCount(0)
    setDistance(0)
    onRecordingStateChangeRef.current?.(false)
    resetViewRef.current?.()
  }

  useImperativeHandle(ref, () => ({
    finalizeRecording: finishRecording,
    resetRecording: resetCamera,
  }))

  return (
    <section className="camera-recorder">
      <div className="camera-viewport" ref={mountRef}>
        <span
          className={[
            'camera-viewport-badge',
            recording ? 'recording' : '',
          ].filter(Boolean).join(' ')}
        >
          <i className={recording ? 'recording' : ''} />
          {recording ? 'RECORDING · 24 FPS' : '24 FPS · MOVE TO START RECORDING'}
        </span>
        <button
          type="button"
          className="camera-reset-button"
          onClick={resetCamera}
          aria-label="Reset camera"
          title="Reset camera"
        >
          <RotateCcw size={12} />
          Reset
        </button>
        <div className="camera-action-display camera-action-left" aria-label="Movement keys">
          <span className="camera-action-label">MOVE</span>
          <div className="camera-key-group camera-key-wasd">
            <kbd className={activeKeys.has('KeyW') ? 'active' : ''}>W</kbd>
            <kbd className={activeKeys.has('KeyA') ? 'active' : ''}>A</kbd>
            <kbd className={activeKeys.has('KeyS') ? 'active' : ''}>S</kbd>
            <kbd className={activeKeys.has('KeyD') ? 'active' : ''}>D</kbd>
          </div>
        </div>
        <div className="camera-action-display camera-action-right" aria-label="Rotation keys">
          <span className="camera-action-label">LOOK</span>
          <div className="camera-key-group camera-key-arrows">
            <kbd className={activeKeys.has('ArrowUp') ? 'active' : ''}>↑</kbd>
            <kbd className={activeKeys.has('ArrowLeft') ? 'active' : ''}>←</kbd>
            <kbd className={activeKeys.has('ArrowDown') ? 'active' : ''}>↓</kbd>
            <kbd className={activeKeys.has('ArrowRight') ? 'active' : ''}>→</kbd>
          </div>
        </div>
        {viewerAvailable ? (
          <span className="camera-focus-hint">
            right-drag to look · tune motion gain below
          </span>
        ) : (
          <span className="camera-viewer-unavailable">
            3D recording is unavailable in this browser. Import a JSON path instead.
          </span>
        )}
      </div>
      <div className="camera-stats">
        <span><small>DURATION</small>{formatDuration(durationMs)}</span>
        <span><small>FRAMES</small>{frameCount}</span>
        <span><small>RATE</small>{SAMPLE_RATE} FPS</span>
        <span><small>DISTANCE</small>{distance.toFixed(2)} m</span>
      </div>
      <div className="camera-scale-controls">
        <label>
          <span>
            <strong>Translation</strong>
            <small>Movement</small>
          </span>
          <span className="camera-scale-input">
            <input
              type="number"
              min="0.1"
              step="0.1"
              disabled={disabled || translationScale === undefined}
              value={translationScaleInput}
              onFocus={(event) => event.currentTarget.select()}
              onChange={(event) => {
                const input = event.target.value
                setTranslationScaleInput(input)
                const value = Number(input)
                if (input !== '' && Number.isFinite(value) && value > 0) {
                  onTranslationScaleChange(value)
                }
              }}
              onBlur={() => {
                setTranslationScaleInput(translationScale === undefined ? '' : String(translationScale))
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter') event.currentTarget.blur()
              }}
              aria-label="Camera translation scale"
            />
            <i>×</i>
          </span>
        </label>
        <label>
          <span>
            <strong>Rotation</strong>
            <small>Yaw, pitch &amp; roll · preview 0.75×</small>
          </span>
          <span className="camera-scale-input">
            <input
              type="number"
              min="0.1"
              step="0.1"
              disabled={disabled || rotationScale === undefined}
              value={rotationScaleInput}
              onFocus={(event) => event.currentTarget.select()}
              onChange={(event) => {
                const input = event.target.value
                setRotationScaleInput(input)
                const value = Number(input)
                if (input !== '' && Number.isFinite(value) && value > 0) {
                  onRotationScaleChange(value)
                }
              }}
              onBlur={() => {
                setRotationScaleInput(rotationScale === undefined ? '' : String(rotationScale))
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter') event.currentTarget.blur()
              }}
              aria-label="Camera rotation scale"
            />
            <i>×</i>
          </span>
        </label>
      </div>
    </section>
  )
})

export const CameraPathDrawer = forwardRef<CameraPathDrawerHandle, CameraPathDrawerProps>(function CameraPathDrawer({
  selected,
  disabled,
  embedded = false,
  theme,
  translationScale,
  rotationScale,
  onTranslationScaleChange,
  onRotationScaleChange,
  onBack,
  onApply,
  onClear,
  onRecordingStateChange,
}: CameraPathDrawerProps, ref) {
  const [draftPath, setDraftPath] = useState<CameraPath | undefined>(selected)
  const [importError, setImportError] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const recorderRef = useRef<CameraRecorderHandle>(null)

  const selectPath = (path: CameraPath) => {
    setDraftPath(path)
    if (embedded) onApply(path)
  }

  useImperativeHandle(
    ref,
    () => ({
      finalizeRecording: () => recorderRef.current?.finalizeRecording() ?? draftPath,
      resetRecording: () => {
        recorderRef.current?.resetRecording()
        setDraftPath(undefined)
        setImportError('')
        onClear()
      },
    }),
    [draftPath, onClear],
  )

  const importPath = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    setImportError('')
    try {
      const raw = JSON.parse(await file.text()) as Record<string, unknown>
      const sampleRate = normalizeSampleRate(raw.sampleRate)
      const frames = normalizeImportedFrames(raw.frames, sampleRate)
      const keyEvents = normalizeImportedKeyEvents(raw.keyEvents)
      if (!frames.length) throw new Error('No valid camera frames found.')
      const name = typeof raw.name === 'string' ? raw.name : file.name.replace(/\.json$/i, '')
      const totalDistance = Number.isFinite(Number(raw.totalDistance))
        ? Number(raw.totalDistance)
        : calculateDistance(frames)
      const rawDurationMs = Number(raw.durationMs)
      const durationMs =
        Number.isFinite(rawDurationMs) && rawDurationMs > 0
          ? rawDurationMs
          : inferredTrajectoryDurationMs(frames, sampleRate)
      // Gain controls now belong to the generation request, not the trajectory.
      delete raw.translationScale
      delete raw.rotationScale
      const payload = {
        ...raw,
        version: typeof raw.version === 'string' ? raw.version : '1.0',
        type: 'camera-trajectory',
        name,
        sampleRate,
        durationMs,
        totalDistance,
        frames,
        keyEvents,
      }
      selectPath({
        id: makeId(),
        name,
        source: 'imported',
        trajectory: JSON.stringify(payload),
        durationMs: Number(payload.durationMs),
        sampleRate: Number(payload.sampleRate),
        totalDistance,
        frames,
        keyEvents,
      })
    } catch (error) {
      setImportError(error instanceof Error ? error.message : 'Could not read this camera path.')
    } finally {
      event.target.value = ''
    }
  }

  const downloadDraft = () => {
    if (!draftPath?.frames?.length) return
    const blob = new Blob([draftPath.trajectory], { type: 'application/json' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = `${draftPath.name.replace(/[\\/:*?"<>|]/g, '-')}.json`
    link.click()
    window.setTimeout(() => URL.revokeObjectURL(link.href), 500)
  }

  return (
    <div className={`camera-drawer${embedded ? ' embedded' : ''}`}>
      {!embedded && (
        <header className="camera-drawer-header">
          <button type="button" className="camera-back-button" onClick={onBack}>
            <ArrowLeft size={15} /> Back
          </button>
          <div>
            <strong>Camera path</strong>
            <small>Pre-plan motion for the next generation</small>
          </div>
        </header>
      )}

      <div className="camera-drawer-scroll">
        <section className={`camera-section${embedded ? ' camera-record-section' : ''}`}>
          <div className="camera-section-title">
            <div>
              <span>CUSTOM</span>
              <strong>Record in 3D</strong>
            </div>
            <button
              type="button"
              className="camera-import-button"
              onClick={() => {
                recorderRef.current?.finalizeRecording()
                fileInputRef.current?.click()
              }}
            >
              <Upload size={13} /> Import JSON
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,application/json"
              aria-label="Camera trajectory JSON"
              hidden
              onChange={importPath}
            />
          </div>
          {importError && <p className="camera-error">{importError}</p>}
          <CameraRecorder
            ref={recorderRef}
            theme={theme}
            disabled={disabled}
            translationScale={translationScale}
            rotationScale={rotationScale}
            onTranslationScaleChange={onTranslationScaleChange}
            onRotationScaleChange={onRotationScaleChange}
            onRecorded={selectPath}
            onRecordingStateChange={onRecordingStateChange}
          />
        </section>

      </div>

      {!embedded && <footer className="camera-drawer-footer">
        <div>
          {draftPath ? (
            <>
              <strong>{draftPath.name}</strong>
              <small>
                {draftPath.source === 'preset'
                  ? 'Preset path · fits output duration'
                  : `${formatDuration(draftPath.durationMs)} · ${draftPath.frames?.length ?? 0} points`}
              </small>
            </>
          ) : (
            <>
              <strong>No camera path</strong>
              <small>The model will choose camera motion</small>
            </>
          )}
        </div>
        {draftPath?.frames?.length ? (
          <button type="button" className="camera-download-button" onClick={downloadDraft} title="Download path">
            <Download size={14} />
          </button>
        ) : null}
        {selected && (
          <button type="button" className="camera-clear-button" onClick={onClear} disabled={disabled}>
            Clear
          </button>
        )}
        <button
          type="button"
          className="camera-apply-button"
          disabled={!draftPath || disabled}
          onClick={() => draftPath && onApply(draftPath)}
        >
          Use path
        </button>
      </footer>}
    </div>
  )
})

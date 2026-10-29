import {
  ArrowUp,
  Bot,
  Camera,
  Check,
  Clapperboard,
  Copy,
  Download,
  Expand,
  Images,
  MoreHorizontal,
  Plus,
  RotateCcw,
  Share2,
  X,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import type {
  BackendConfig,
  CameraPath,
  ChatMessage,
  DryRunJob,
  GenerateJob,
  GalleryItem,
  InferenceSteps,
  PromptEnhancerModel,
  PromptEnhancerReasoningEffort,
  PromptEnhancerReasoningEffortOption,
  PromptEnhancerSeed,
  RollbackJob,
  WorldConfig,
} from '../types'
import { CameraPlaybackOverlay } from './CameraPlaybackOverlay'
import { withFinishedCameraKeyEvents } from '../cameraTrajectory'
import { DemoOverlayTimeline } from './DemoOverlayTimeline'
import { CameraPathDrawer, type CameraPathDrawerHandle } from './CameraPathDrawer'
import { DirectionHud } from './DirectionHud'
import {
  PromptTurnTimeline,
  type PromptTurn,
  type PromptTurnStage,
} from './PromptTurnTimeline'
import { playbackTurnAtTime } from '../playbackTimeline'
import { perspectiveLabel } from '../perspective'
import { buildSharedVideoOverlay } from '../shareOverlay'
import { ResourceStatus } from './ResourceStatus'
import { ThemeToggle } from './ThemeToggle'
import { WorldGallery } from './WorldGallery'

interface WorldStudioProps {
  config: WorldConfig
  backendConfig?: BackendConfig
  dryRun?: boolean
  onReconfigure: () => void
  onSessionExpired: (message?: string) => void
  onSessionReleased?: () => void
  onGalleryItemAdded?: (item: GalleryItem) => void
  promptEnhancerModel?: PromptEnhancerModel
  onPromptEnhancerModelChange?: (model: PromptEnhancerModel) => void
  promptEnhancerReasoningEffort?: PromptEnhancerReasoningEffort
  promptEnhancerReasoningEffortOptions?: PromptEnhancerReasoningEffortOption[]
  onPromptEnhancerReasoningEffortChange?: (
    effort: PromptEnhancerReasoningEffort,
  ) => void
  promptEnhancerSeed?: PromptEnhancerSeed
  promptEnhancerSeedMin?: number
  promptEnhancerSeedMax?: number
  onPromptEnhancerSeedChange?: (
    seed: PromptEnhancerSeed | undefined,
  ) => void
  inferenceSteps?: InferenceSteps
  inferenceStepsOptions?: InferenceSteps[]
  onInferenceStepsChange?: (steps: InferenceSteps) => void
  theme: 'dark' | 'light'
  onThemeToggle: () => void
}

function initialMessages(config: WorldConfig): ChatMessage[] {
  return [
    {
      id: 1,
      role: 'agent',
      content: `Your ${perspectiveLabel(config.perspective).toLowerCase()} world is ready to direct. Use prompts for action, environment, and mood. Camera motion can be planned before each generation.`,
      meta: 'WORLD AGENT',
    },
  ]
}

// Keep completed clips on their finalized MP4 URL. An open MediaSource waits
// for another AR fragment forever, which makes browsers show their native
// buffering spinner after the current segment has already finished.
const USE_MEDIA_SOURCE_STREAMING = false
const DEFAULT_SEGMENT_ESTIMATE_SECONDS = 30
const STATUS_POLL_RETRY_BASE_MS = 750
const STATUS_POLL_RECOVERY_TIMEOUT_MS = 20 * 60 * 1000
export const WORLD_SESSION_IDLE_TIMEOUT_MS = 15 * 60 * 1000
const WORLD_SESSION_STATUS_POLL_MS = 5_000
const WORLD_SESSION_ENDED_MESSAGE =
  'Your world session has ended. Create a new world to continue.'
const EXPLICIT_SESSION_RELEASE_HEADER = 'X-Nomas-Explicit-Session-Release'
const ROLLBACK_CONTROLS_ENABLED = true
type GenerationProgressPhase = 'preparing' | 'queued' | 'rendering'

class TransientGenerationPollError extends Error {}

function isTransientGenerationStatus(status: number): boolean {
  // A dispatcher can briefly lose the submitting host during a Gateway
  // restart and answer 404 from a different slot. Keep polling while the
  // cross-host recovery path catches up instead of marking a real job failed.
  return [404, 408, 423, 425, 429, 500, 502, 503, 504].includes(status)
}

interface GenerationProgressState {
  phase: GenerationProgressPhase
  startedAt?: number
  segmentNumber?: number
  totalSegments?: number
  queuePosition?: number
}

interface GenerateResponse {
  id: string
  status: 'succeeded'
  video_url: string
  frames: number
  fps: number
  generation_seconds: number
  playback_start_seconds?: number
  segment_frames?: number
  stream_init_url?: string
  stream_segment_url?: string
  stream_sequence?: number
  stream_mime_type?: string
  stream_replace_start_seconds?: number
  prompt_enhanced?: boolean
}

interface ShareResponse {
  id: string
  share_path: string
}

interface DownloadResponse {
  download_path: string
  filename: string
  overlay: boolean
}

interface GalleryResponse {
  item: GalleryItem
  added: boolean
}

interface SegmentTiming {
  start_time?: string | number
  end_time?: string | number
  contained_actions?: SegmentTiming[] | SegmentTiming
}

interface StructuredPrompt {
  segments: SegmentTiming[]
}

function cameraPathDurationMs(path: CameraPath): number | undefined {
  const explicitDurationMs = Number(path.durationMs)
  if (Number.isFinite(explicitDurationMs) && explicitDurationMs > 0) {
    return explicitDurationMs
  }
  const frames = path.frames ?? []
  if (!frames.length) return undefined
  const sampleRate =
    Number.isFinite(Number(path.sampleRate)) && Number(path.sampleRate) > 0
      ? Number(path.sampleRate)
      : 24
  const intervalMs = 1000 / sampleRate
  const firstTimestamp = Number(frames[0]?.t)
  const lastTimestamp = Number(frames.at(-1)?.t)
  const timestampDuration =
    Number.isFinite(firstTimestamp)
    && Number.isFinite(lastTimestamp)
    && lastTimestamp >= firstTimestamp
      ? lastTimestamp - firstTimestamp + intervalMs
      : 0
  return Math.max(intervalMs, frames.length * intervalMs, timestampDuration)
}

function parseTimestamp(value: string | number): number {
  if (typeof value === 'number') return value
  const parts = value.trim().split(':').map(Number)
  if (parts.some((part) => !Number.isFinite(part))) {
    throw new Error(`Invalid timestamp: ${value}`)
  }
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2]
  if (parts.length === 2) return parts[0] * 60 + parts[1]
  if (parts.length === 1) return parts[0]
  throw new Error(`Invalid timestamp: ${value}`)
}

function segmentDuration(segment: SegmentTiming): number {
  const actions = Array.isArray(segment.contained_actions)
    ? segment.contained_actions
    : segment.contained_actions
      ? [segment.contained_actions]
      : []
  if (segment.end_time !== undefined) {
    const start =
      segment.start_time !== undefined ? parseTimestamp(segment.start_time) : 0
    const duration = parseTimestamp(segment.end_time) - start
    if (duration <= 0) throw new Error(`Invalid segment duration: ${duration}`)
    return duration
  }

  const starts: number[] = []
  const ends: number[] = []
  for (const action of actions) {
    if (action.start_time !== undefined) starts.push(parseTimestamp(action.start_time))
    if (action.end_time !== undefined) ends.push(parseTimestamp(action.end_time))
  }
  if (ends.length === 0) throw new Error('Each segment needs an end_time')

  const duration = Math.max(...ends) - (starts.length > 0 ? Math.min(...starts) : 0)
  if (duration <= 0) throw new Error(`Invalid segment duration: ${duration}`)
  return duration
}

function parseSegArJson(
  raw: string,
  fps: number,
): {
  prompt: string
  num_frames: number
  segment_boundaries: number[]
} | undefined {
  const trimmed = raw.trim()
  if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) return undefined

  const parsed = JSON.parse(trimmed) as StructuredPrompt | StructuredPrompt[]
  const item = Array.isArray(parsed) ? parsed[0] : parsed
  if (!item || !Array.isArray(item.segments) || item.segments.length === 0) {
    throw new Error('Structured seg_ar JSON requires a non-empty segments array')
  }

  const prompts = item.segments.map((segment, index) => {
    const promptItem =
      index === 0
        ? { ...item, segments: [segment] }
        : { segments: [segment] }
    return JSON.stringify(promptItem)
  })

  const boundaries = [0]
  let cumulativeSeconds = 0
  for (const segment of item.segments) {
    cumulativeSeconds += segmentDuration(segment)
    boundaries.push(
      Math.max(
        boundaries[boundaries.length - 1] + 1,
        Math.floor(cumulativeSeconds * fps),
      ),
    )
  }

  return {
    prompt: prompts.join('|||'),
    num_frames: boundaries[boundaries.length - 1],
    segment_boundaries: boundaries,
  }
}

function chooseStructuredSegArInput(direction: string, subject: string): string {
  const trimmedDirection = direction.trim()
  if (trimmedDirection.startsWith('{') || trimmedDirection.startsWith('[')) {
    return trimmedDirection
  }
  const trimmedSubject = subject.trim()
  if (trimmedSubject.startsWith('{') || trimmedSubject.startsWith('[')) {
    return trimmedSubject
  }
  return trimmedDirection
}

function formatPlaybackTime(seconds: number): string {
  const safeSeconds = Number.isFinite(seconds) && seconds > 0
    ? Math.floor(seconds)
    : 0
  const hours = Math.floor(safeSeconds / 3600)
  const minutes = Math.floor((safeSeconds % 3600) / 60)
  const remainder = safeSeconds % 60
  return [hours, minutes, remainder]
    .map((part) => String(part).padStart(2, '0'))
    .join(':')
}

export function WorldStudio({
  config,
  backendConfig,
  dryRun = false,
  onReconfigure,
  onSessionExpired,
  onSessionReleased,
  onGalleryItemAdded,
  promptEnhancerModel = 'Nomas-PE',
  onPromptEnhancerModelChange = () => {},
  promptEnhancerReasoningEffort = 'high',
  promptEnhancerReasoningEffortOptions = [
    { id: 'low', label: 'Low' },
    { id: 'medium', label: 'Medium' },
    { id: 'high', label: 'High', default: true },
  ],
  onPromptEnhancerReasoningEffortChange = () => undefined,
  promptEnhancerSeed,
  promptEnhancerSeedMin = 0,
  promptEnhancerSeedMax = 2 ** 31 - 1,
  onPromptEnhancerSeedChange = () => undefined,
  inferenceSteps = 8,
  inferenceStepsOptions = [4, 6, 8],
  onInferenceStepsChange = () => undefined,
  theme,
  onThemeToggle,
}: WorldStudioProps) {
  const [messages, setMessages] = useState<ChatMessage[]>(() => initialMessages(config))
  const [draft, setDraft] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [videoUrl, setVideoUrl] = useState<string>()
  const [videoCurrentTime, setVideoCurrentTime] = useState(0)
  const [videoDuration, setVideoDuration] = useState(0)
  const [videoResolution, setVideoResolution] = useState('')
  const [transitionFrame, setTransitionFrame] = useState<string>()
  const [transitionFrameVisible, setTransitionFrameVisible] = useState(false)
  const [downloadUrl, setDownloadUrl] = useState<string>()
  const [generationError, setGenerationError] = useState<string>()
  const [cameraPath, setCameraPath] = useState<CameraPath>()
  const [cameraScaleOverrides, setCameraScaleOverrides] = useState<{
    translation?: number
    rotation?: number
  }>({})
  const translationScale = cameraScaleOverrides.translation
    ?? backendConfig?.trajectory_translation_scale
  const rotationScale = cameraScaleOverrides.rotation
    ?? backendConfig?.trajectory_rotation_scale
  const [isCameraRecording, setIsCameraRecording] = useState(false)
  const [inputMode, setInputMode] = useState<'prompt' | 'camera'>('prompt')
  const [moreMenuOpen, setMoreMenuOpen] = useState(false)
  const [userPromptVisible, setUserPromptVisible] = useState(true)
  const [shareUrl, setShareUrl] = useState<string>()
  const [shareError, setShareError] = useState<string>()
  const [isSharing, setIsSharing] = useState(false)
  const [shareCopied, setShareCopied] = useState(false)
  const [isAddingToGallery, setIsAddingToGallery] = useState(false)
  const [galleryAdded, setGalleryAdded] = useState(false)
  const [galleryError, setGalleryError] = useState<string>()
  const [galleryOpen, setGalleryOpen] = useState(false)
  const [downloadDialogOpen, setDownloadDialogOpen] = useState(false)
  const [downloadMode, setDownloadMode] =
    useState<'overlay' | 'original'>('overlay')
  const [downloadError, setDownloadError] = useState<string>()
  const [isPreparingDownload, setIsPreparingDownload] = useState(false)
  const [overlayDownloadStatus, setOverlayDownloadStatus] =
    useState<'idle' | 'preparing' | 'ready' | 'failed'>('idle')
  const [isFullscreen, setIsFullscreen] = useState(false)
  const overlayDownloadRequestRef = useRef<{
    key: string
    request: Promise<DownloadResponse>
  } | undefined>(undefined)
  const moreMenuRef = useRef<HTMLDivElement>(null)
  const cameraDrawerRef = useRef<CameraPathDrawerHandle>(null)
  const videoFrameRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const nextMessageId = useRef(10)
  const dryRunStartedRef = useRef(false)
  const initialActionStartedRef = useRef(false)
  const lastDryRunSegmentRef = useRef(0)
  const pendingResumeTimeRef = useRef<number | undefined>(undefined)
  const lastFrameSnapshotRef = useRef<{
    source: string
    dataUrl: string
  } | undefined>(undefined)
  const transitionFrameActiveRef = useRef(false)
  const transitionCleanupTimerRef = useRef<number | undefined>(undefined)
  const mediaSourceRef = useRef<MediaSource | undefined>(undefined)
  const sourceBufferRef = useRef<SourceBuffer | undefined>(undefined)
  const mediaSourceUrlRef = useRef<string | undefined>(undefined)
  const streamInitUrlRef = useRef<string | undefined>(undefined)
  const streamSequenceRef = useRef(-1)
  const streamAppendRef = useRef<Promise<void>>(Promise.resolve())
  const mseFailedRef = useRef(false)
  const sessionIdRef = useRef(
    globalThis.crypto?.randomUUID?.()
      ?? `nomas-${Date.now()}-${Math.random().toString(36).slice(2)}`,
  )
  const firstSessionRequestRef = useRef(true)
  const replacingRolledBackFirstSegmentRef = useRef(false)
  const isOneShot = backendConfig?.rollout_type === 'oneshot'
  const [dryRunJob, setDryRunJob] = useState<DryRunJob>()
  const [interactiveSegmentCount, setInteractiveSegmentCount] = useState(0)
  const [promptTurns, setPromptTurns] = useState<PromptTurn[]>([])
  const [generationProgress, setGenerationProgress] =
    useState<GenerationProgressState>({ phase: 'preparing' })
  const [progressClock, setProgressClock] = useState(() => Date.now())
  const completedTurnIdsRef = useRef<number[]>([])
  const playingTurnIdRef = useRef<number | undefined>(undefined)
  const segmentEstimateSecondsRef = useRef(DEFAULT_SEGMENT_ESTIMATE_SECONDS)
  const onSessionExpiredRef = useRef(onSessionExpired)
  const inferenceModelLabel = backendConfig?.model_name ?? '—'

  useEffect(() => {
    onSessionExpiredRef.current = onSessionExpired
  }, [onSessionExpired])

  useEffect(() => {
    if (dryRun || isGenerating) return

    let timer = window.setTimeout(
      () => onSessionExpiredRef.current(),
      WORLD_SESSION_IDLE_TIMEOUT_MS,
    )
    const resetIdleTimer = () => {
      window.clearTimeout(timer)
      timer = window.setTimeout(
        () => onSessionExpiredRef.current(),
        WORLD_SESSION_IDLE_TIMEOUT_MS,
      )
    }
    const activityEvents: Array<keyof WindowEventMap> = [
      'pointerdown',
      'keydown',
      'touchstart',
      'wheel',
    ]
    activityEvents.forEach((eventName) => {
      window.addEventListener(eventName, resetIdleTimer, { passive: true })
    })
    return () => {
      window.clearTimeout(timer)
      activityEvents.forEach((eventName) => {
        window.removeEventListener(eventName, resetIdleTimer)
      })
    }
  }, [dryRun, isGenerating])

  useEffect(() => {
    if (dryRun || isOneShot || isGenerating || interactiveSegmentCount === 0) return

    const controller = new AbortController()
    let checking = false
    let expired = false
    const checkSession = async () => {
      if (checking || expired || controller.signal.aborted) return
      checking = true
      try {
        const response = await fetch(
          `${import.meta.env.BASE_URL}v1/sessions/${encodeURIComponent(sessionIdRef.current)}/status`,
          { cache: 'no-store', signal: controller.signal },
        )
        if (!response.ok) return
        const status = await response.json() as { status?: string }
        if (!controller.signal.aborted && status.status === 'expired') {
          expired = true
          onSessionExpiredRef.current(WORLD_SESSION_ENDED_MESSAGE)
        }
      } catch {
        // A transient network failure does not mean the world has expired.
      } finally {
        checking = false
      }
    }
    const checkVisibleSession = () => {
      if (document.visibilityState === 'visible') void checkSession()
    }
    void checkSession()
    const timer = window.setInterval(() => void checkSession(), WORLD_SESSION_STATUS_POLL_MS)
    window.addEventListener('focus', checkVisibleSession)
    window.addEventListener('pageshow', checkVisibleSession)
    document.addEventListener('visibilitychange', checkVisibleSession)
    return () => {
      controller.abort()
      window.clearInterval(timer)
      window.removeEventListener('focus', checkVisibleSession)
      window.removeEventListener('pageshow', checkVisibleSession)
      document.removeEventListener('visibilitychange', checkVisibleSession)
    }
  }, [dryRun, isGenerating, isOneShot, interactiveSegmentCount])

  useEffect(() => {
    const sessionId = sessionIdRef.current
    let released = false
    let pageHiding = false
    const release = () => {
      if (released) return
      released = true
      void fetch(
        `${import.meta.env.BASE_URL}v1/sessions/${encodeURIComponent(sessionId)}/release`,
        {
          method: 'POST',
          keepalive: true,
          headers: {
            [EXPLICIT_SESSION_RELEASE_HEADER]: '1',
          },
        },
      )
        .catch(() => undefined)
        .finally(() => onSessionReleased?.())
    }
    const preserveSessionForPageHide = () => {
      pageHiding = true
    }
    const resumeAfterPageShow = () => {
      pageHiding = false
    }
    window.addEventListener('pagehide', preserveSessionForPageHide)
    window.addEventListener('pageshow', resumeAfterPageShow)
    return () => {
      window.removeEventListener('pagehide', preserveSessionForPageHide)
      window.removeEventListener('pageshow', resumeAfterPageShow)
      if (!pageHiding) release()
    }
  }, [])

  useEffect(() => {
    if (!isGenerating) return
    setProgressClock(Date.now())
    const timer = window.setInterval(() => setProgressClock(Date.now()), 250)
    return () => window.clearInterval(timer)
  }, [isGenerating])

  useEffect(() => {
    if (!moreMenuOpen) return

    const closeOnOutsideClick = (event: PointerEvent) => {
      if (
        event.target instanceof Node
        && !moreMenuRef.current?.contains(event.target)
      ) {
        setMoreMenuOpen(false)
      }
    }
    const closeOnEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') setMoreMenuOpen(false)
    }

    document.addEventListener('pointerdown', closeOnOutsideClick)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('pointerdown', closeOnOutsideClick)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [moreMenuOpen])

  useEffect(() => {
    const updateFullscreenState = () => {
      setIsFullscreen(document.fullscreenElement === videoFrameRef.current)
    }
    document.addEventListener('fullscreenchange', updateFullscreenState)
    return () => {
      document.removeEventListener('fullscreenchange', updateFullscreenState)
    }
  }, [])

  const progressStartedAt = generationProgress.startedAt
  const segmentElapsedSeconds =
    progressStartedAt === undefined
      ? 0
      : Math.max(0, (progressClock - progressStartedAt) / 1000)
  const segmentEstimateSeconds = segmentEstimateSecondsRef.current
  const segmentProgressPercent =
    generationProgress.phase === 'rendering'
      ? Math.min(
          95,
          Math.max(2, (segmentElapsedSeconds / segmentEstimateSeconds) * 100),
        )
      : 0
  const segmentRemainingSeconds = Math.max(
    0,
    Math.ceil(segmentEstimateSeconds - segmentElapsedSeconds),
  )
  const progressLabel =
    generationProgress.phase === 'preparing'
      ? 'PREPARING SEGMENT'
      : generationProgress.phase === 'queued'
        ? generationProgress.queuePosition
          ? `QUEUED · ${generationProgress.queuePosition} AHEAD`
          : 'WAITING FOR GPU'
        : generationProgress.segmentNumber && generationProgress.totalSegments
          ? `SEGMENT ${generationProgress.segmentNumber}/${generationProgress.totalSegments}`
          : 'CURRENT SEGMENT'
  const progressEta =
    generationProgress.phase === 'rendering'
      ? segmentRemainingSeconds > 0
        ? `~${segmentRemainingSeconds}s left`
        : 'Finishing…'
      : 'Estimating…'
  const playbackTurn = useMemo(() => {
    return playbackTurnAtTime(
      promptTurns,
      videoCurrentTime,
      playingTurnIdRef.current,
    )
  }, [promptTurns, videoCurrentTime])

  const updatePromptTurn = (
    id: number,
    update: Partial<PromptTurn> | ((turn: PromptTurn) => Partial<PromptTurn>),
  ) => {
    setPromptTurns((current) =>
      current.map((turn) =>
        turn.id === id
          ? { ...turn, ...(typeof update === 'function' ? update(turn) : update) }
          : turn,
      ),
    )
  }

  const setPromptTurnStage = (id: number, stage: PromptTurnStage) => {
    updatePromptTurn(id, { stage })
  }

  const captureVideoFrame = (video: HTMLVideoElement) => {
    if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
      return undefined
    }
    try {
      const canvas = document.createElement('canvas')
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const context = canvas.getContext('2d')
      if (!context) return undefined
      context.drawImage(video, 0, 0, canvas.width, canvas.height)
      return canvas.toDataURL('image/jpeg', 0.92)
    } catch (error) {
      console.warn('[Hy-Nomas] Could not capture the current video frame', error)
      return undefined
    }
  }

  const rememberLastVideoFrame = (video: HTMLVideoElement) => {
    const dataUrl = captureVideoFrame(video)
    if (!dataUrl || !videoUrl) return
    lastFrameSnapshotRef.current = {
      source: videoUrl,
      dataUrl,
    }
  }

  const holdPreviousVideoFrame = () => {
    const video = videoRef.current
    if (!videoUrl || !video) return false
    const remembered =
      lastFrameSnapshotRef.current?.source === videoUrl
        ? lastFrameSnapshotRef.current.dataUrl
        : undefined
    const dataUrl = remembered ?? captureVideoFrame(video)
    if (!dataUrl) return false
    if (transitionCleanupTimerRef.current !== undefined) {
      window.clearTimeout(transitionCleanupTimerRef.current)
    }
    transitionFrameActiveRef.current = true
    setTransitionFrame(dataUrl)
    setTransitionFrameVisible(true)
    return true
  }

  const releasePreviousVideoFrame = () => {
    if (!transitionFrameActiveRef.current) return
    transitionFrameActiveRef.current = false
    setTransitionFrameVisible(false)
    if (transitionCleanupTimerRef.current !== undefined) {
      window.clearTimeout(transitionCleanupTimerRef.current)
    }
    transitionCleanupTimerRef.current = window.setTimeout(() => {
      setTransitionFrame(undefined)
      transitionCleanupTimerRef.current = undefined
    }, 260)
  }

  const appendBuffer = (sourceBuffer: SourceBuffer, bytes: ArrayBuffer) =>
    new Promise<void>((resolve, reject) => {
      const cleanup = () => {
        sourceBuffer.removeEventListener('updateend', onUpdateEnd)
        sourceBuffer.removeEventListener('error', onError)
      }
      const onUpdateEnd = () => {
        cleanup()
        resolve()
      }
      const onError = () => {
        cleanup()
        reject(new Error('MediaSource failed to append a video fragment'))
      }
      sourceBuffer.addEventListener('updateend', onUpdateEnd, { once: true })
      sourceBuffer.addEventListener('error', onError, { once: true })
      sourceBuffer.appendBuffer(bytes)
    })

  const fetchBytes = async (relativeUrl: string) => {
    const response = await fetch(`${import.meta.env.BASE_URL}${relativeUrl}`, {
      cache: 'no-store',
    })
    if (!response.ok) {
      throw new Error(`Failed to fetch video fragment: HTTP ${response.status}`)
    }
    return response.arrayBuffer()
  }

  const resetMediaStream = () => {
    sourceBufferRef.current = undefined
    mediaSourceRef.current = undefined
    streamInitUrlRef.current = undefined
    streamSequenceRef.current = -1
    streamAppendRef.current = Promise.resolve()
    mseFailedRef.current = false
    if (mediaSourceUrlRef.current) {
      URL.revokeObjectURL(mediaSourceUrlRef.current)
      mediaSourceUrlRef.current = undefined
    }
  }

  const appendStreamSegment = async (result: GenerateResponse): Promise<boolean> => {
    const {
      stream_init_url: initUrl,
      stream_segment_url: segmentUrl,
      stream_sequence: sequence,
      stream_mime_type: mimeType,
    } = result
    if (
      !initUrl
      || !segmentUrl
      || sequence === undefined
      || !mimeType
      || mseFailedRef.current
      || typeof MediaSource === 'undefined'
      || !MediaSource.isTypeSupported(mimeType)
    ) {
      return false
    }

    streamAppendRef.current = streamAppendRef.current.then(async () => {
      const needsNewStream =
        !mediaSourceRef.current
        || !sourceBufferRef.current
        || streamInitUrlRef.current !== initUrl
        || sequence === 0
      if (needsNewStream) {
        if (mediaSourceUrlRef.current) {
          URL.revokeObjectURL(mediaSourceUrlRef.current)
        }
        const mediaSource = new MediaSource()
        const objectUrl = URL.createObjectURL(mediaSource)
        mediaSourceRef.current = mediaSource
        sourceBufferRef.current = undefined
        mediaSourceUrlRef.current = objectUrl
        streamInitUrlRef.current = initUrl
        streamSequenceRef.current = -1
        setVideoUrl(objectUrl)
        await new Promise<void>((resolve, reject) => {
          mediaSource.addEventListener('sourceopen', () => resolve(), { once: true })
          mediaSource.addEventListener(
            'sourceclose',
            () => reject(new Error('MediaSource closed before initialization')),
            { once: true },
          )
        })
        const sourceBuffer = mediaSource.addSourceBuffer(mimeType)
        sourceBuffer.mode = 'sequence'
        sourceBufferRef.current = sourceBuffer
        await appendBuffer(sourceBuffer, await fetchBytes(initUrl))
      }

      if (sequence <= streamSequenceRef.current) return
      if (sequence !== streamSequenceRef.current + 1) {
        throw new Error(
          `Out-of-order video fragment: expected ${streamSequenceRef.current + 1}, received ${sequence}`,
        )
      }
      const video = videoRef.current
      const previousEnd = result.playback_start_seconds ?? 0
      const firstSegment = sequence === 0
      const replaceStart = result.stream_replace_start_seconds
      const replacingTail =
        typeof replaceStart === 'number' && Number.isFinite(replaceStart)
      const waitingAtBoundary = Boolean(
        video
        && (
          video.ended
          || video.currentTime >= Math.max(previousEnd - 0.25, 0)
        ),
      )
      const wasPlaying = Boolean(video && !video.paused)
      const sourceBuffer = sourceBufferRef.current
      if (!sourceBuffer) throw new Error('MediaSource buffer is unavailable')
      if (replacingTail) {
        const bufferedEnd =
          sourceBuffer.buffered.length > 0
            ? sourceBuffer.buffered.end(sourceBuffer.buffered.length - 1)
            : 0
        const removeStart = Math.max(0, replaceStart)
        if (bufferedEnd > removeStart) {
          sourceBuffer.abort()
          await new Promise<void>((resolve, reject) => {
            const cleanup = () => {
              sourceBuffer.removeEventListener('updateend', onUpdateEnd)
              sourceBuffer.removeEventListener('error', onError)
            }
            const onUpdateEnd = () => {
              cleanup()
              resolve()
            }
            const onError = () => {
              cleanup()
              reject(new Error('MediaSource failed to replace the PixelUMM tail'))
            }
            sourceBuffer.addEventListener('updateend', onUpdateEnd, { once: true })
            sourceBuffer.addEventListener('error', onError, { once: true })
            sourceBuffer.remove(removeStart, bufferedEnd)
          })
        }
        // The replacement fragment starts at the old provisional tail rather
        // than after the previous fragment.
        sourceBuffer.timestampOffset = removeStart
      }
      await appendBuffer(sourceBuffer, await fetchBytes(segmentUrl))
      streamSequenceRef.current = sequence
      const mediaSource = mediaSourceRef.current
      if (mediaSource?.readyState === 'open') {
        const bufferedEnd =
          sourceBuffer.buffered.length > 0
            ? sourceBuffer.buffered.end(sourceBuffer.buffered.length - 1)
            : 0
        // An open MediaSource has no natural EOF because more AR segments may
        // arrive. Pin its visible duration to the known cumulative frame count
        // instead of letting controls floor the last frame PTS (e.g. 4.958s)
        // to 0:04 for a 120-frame / 24-FPS segment.
        mediaSource.duration = Math.max(
          result.frames / result.fps,
          bufferedEnd,
        )
      }
      if (video && (firstSegment || waitingAtBoundary)) {
        if (!firstSegment) {
          video.currentTime = previousEnd
        }
        await video.play().catch(() => {
          // Browser autoplay policy may require an explicit click.
        })
      } else if (video && wasPlaying) {
        // Appending does not replace src, so active replay continues naturally.
      }
    })
    try {
      await streamAppendRef.current
      return true
    } catch (error) {
      console.warn('[Hy-Nomas] MSE append failed; falling back to cumulative MP4', error)
      mseFailedRef.current = true
      sourceBufferRef.current = undefined
      mediaSourceRef.current = undefined
      streamInitUrlRef.current = undefined
      streamSequenceRef.current = -1
      if (mediaSourceUrlRef.current) {
        URL.revokeObjectURL(mediaSourceUrlRef.current)
        mediaSourceUrlRef.current = undefined
      }
      return false
    }
  }

  useEffect(
    () => () => {
      if (mediaSourceUrlRef.current) {
        URL.revokeObjectURL(mediaSourceUrlRef.current)
      }
      if (transitionCleanupTimerRef.current !== undefined) {
        window.clearTimeout(transitionCleanupTimerRef.current)
      }
    },
    [],
  )

  const describedSubject = useMemo(
    () => config.subjectDescription.trim(),
    [config.subjectDescription],
  )

  useEffect(() => {
    if (!dryRun) return
    if (dryRunStartedRef.current) return
    dryRunStartedRef.current = true
    let cancelled = false
    let timer: number | undefined

    const poll = async (jobId: string) => {
      const response = await fetch(`${import.meta.env.BASE_URL}v1/dry-run/${jobId}`, {
        cache: 'no-store',
      })
      const payload = (await response.json()) as DryRunJob | { detail?: string }
      if (!response.ok) {
        throw new Error('detail' in payload && payload.detail ? payload.detail : `HTTP ${response.status}`)
      }
      if (cancelled) return
      const job = payload as DryRunJob
      setDryRunJob(job)
      setIsGenerating(job.status === 'queued' || job.status === 'running')
      setGenerationProgress((current) => {
        if (job.status === 'queued') {
          return {
            phase: 'queued',
            queuePosition: job.queue_position,
            segmentNumber: Math.min(job.completed_segments + 1, job.total_segments),
            totalSegments: job.total_segments,
          }
        }
        if (job.status !== 'running') return current
        const segmentNumber = Math.min(
          job.completed_segments + 1,
          job.total_segments,
        )
        const sameSegment =
          current.phase === 'rendering'
          && current.segmentNumber === segmentNumber
          && current.totalSegments === job.total_segments
        if (!sameSegment && current.phase === 'rendering' && current.startedAt) {
          const elapsed = Math.max(1, (Date.now() - current.startedAt) / 1000)
          segmentEstimateSecondsRef.current = Math.max(
            8,
            Math.min(180, segmentEstimateSecondsRef.current * 0.65 + elapsed * 0.35),
          )
        }
        return {
          phase: 'rendering',
          startedAt: sameSegment ? current.startedAt : Date.now(),
          segmentNumber,
          totalSegments: job.total_segments,
        }
      })
      if (job.video_url && job.completed_segments > lastDryRunSegmentRef.current) {
        const latestUpdate = job.segment_updates.at(-1)
        pendingResumeTimeRef.current =
          latestUpdate?.playback_start_seconds
          ?? (latestUpdate?.start_frame !== undefined
            ? latestUpdate.start_frame / (backendConfig?.fps ?? 24)
            : 0)
        lastDryRunSegmentRef.current = job.completed_segments
        setVideoUrl(`${job.video_url}?segment=${job.completed_segments}`)
      }
      const visibleCaptionCount =
        job.status === 'queued' || job.status === 'running'
          ? Math.min(job.total_segments, job.completed_segments + 1)
          : job.completed_segments
      setMessages([
        {
          id: 1,
          role: 'agent',
          content:
            job.status === 'queued'
              ? `Waiting for the GPU worker${job.queue_position ? ` · ${job.queue_position} job(s) ahead` : ''}…`
              : `Dry run sample: ${job.sample_id}`,
          meta:
            job.status === 'queued'
              ? 'QUEUED'
              : `${job.completed_segments}/${job.total_segments} SEGMENTS`,
        },
        ...job.captions.slice(0, visibleCaptionCount).map((caption, segmentIndex) => ({
          id: segmentIndex + 10,
          role: 'agent' as const,
          content: caption,
          meta:
            segmentIndex < job.completed_segments
              ? `SEGMENT ${segmentIndex + 1} · DECODED`
              : `SEGMENT ${segmentIndex + 1} · GENERATING`,
        })),
        ...(job.status === 'failed'
          ? [{
              id: 9999,
              role: 'agent' as const,
              content: `Dry run failed: ${job.error ?? 'unknown error'}`,
              meta: 'FAILED',
            }]
          : []),
      ])
      if (job.status === 'queued' || job.status === 'running') {
        timer = window.setTimeout(() => void poll(jobId).catch(handleError), 1500)
      }
    }

    const handleError = (error: unknown) => {
      if (cancelled) return
      const message = error instanceof Error ? error.message : String(error)
      setGenerationError(message)
      setIsGenerating(false)
    }

    setIsGenerating(true)
    setGenerationProgress({ phase: 'preparing' })
    setMessages([{
      id: 1,
      role: 'agent',
      content: 'Selecting a random benchmark sample and starting segment-autoregressive inference…',
      meta: 'DRY RUN',
    }])
    void fetch(`${import.meta.env.BASE_URL}v1/dry-run`, { method: 'POST' })
      .then(async (response) => {
        const payload = (await response.json()) as DryRunJob | { detail?: string }
        if (!response.ok) {
          throw new Error('detail' in payload && payload.detail ? payload.detail : `HTTP ${response.status}`)
        }
        const job = payload as DryRunJob
        if (!cancelled) {
          setDryRunJob(job)
          return poll(job.id)
        }
      })
      .catch(handleError)

    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [dryRun])

  const buildPrompt = (action: string) => {
    const direction = action.trim() || 'Keep the current world unchanged and follow the provided camera trajectory.'
    return [
      `Perspective: ${config.perspective}.`,
      `Subject description: ${describedSubject}.`,
      `Initial environment: ${config.environmentDescription.trim()}.`,
      `Current direction: ${direction}.`,
    ].join(' ')
  }

  const sendPrompt = async (text = draft, cameraOverride?: CameraPath) => {
    const prompt = text.trim()
    const selectedCameraPath = inputMode === 'camera' ? cameraOverride ?? cameraPath : undefined
    const activeCameraPath = selectedCameraPath
      ? withFinishedCameraKeyEvents(selectedCameraPath)
      : undefined
    if ((inputMode === 'prompt' ? !prompt : !activeCameraPath) || isGenerating) return
    const messageContent = inputMode === 'prompt' ? prompt : 'Generate from the selected camera path.'

    const id = nextMessageId.current
    nextMessageId.current += 3
    const turnNumber = promptTurns.length + 1
    setPromptTurns((current) => [
      ...current,
      {
        id,
        number: current.length + 1,
        prompt: messageContent,
        submittedAt: new Date().toLocaleTimeString('en-GB', { hour12: false }),
        stage: 'simulating',
        cameraLabel: activeCameraPath?.name,
        cameraPath: activeCameraPath ? {
          ...activeCameraPath,
          visualizationRotationScale: rotationScale ?? 1,
        } : undefined,
      },
    ])
    setMessages((current) => [
      ...current,
      {
        id,
        role: 'user',
        content: messageContent,
        meta: `PROMPT ${turnNumber}`,
        cameraLabel: activeCameraPath?.name,
      },
      {
        id: id + 1,
        role: 'agent',
        content: 'Generating a new segment-autoregressive world preview…',
        meta: 'GENERATING',
      },
    ])
    setDraft('')
    setGenerationError(undefined)
    setIsGenerating(true)
    setGenerationProgress({ phase: 'preparing' })

    try {
      const requestUrl = `${import.meta.env.BASE_URL}v1/generate`
      const cameraDurationMs = activeCameraPath
        ? cameraPathDurationMs(activeCameraPath)
        : undefined
      const structuredSegAr = isOneShot
        || activeCameraPath
        ? undefined
        : parseSegArJson(
            chooseStructuredSegArInput(prompt, config.subjectDescription),
            backendConfig?.fps ?? 24,
          )
      const segmentCount = structuredSegAr
        ? structuredSegAr.segment_boundaries.length - 1
        : 1
      const initialEnvironmentDescription = (
        replacingRolledBackFirstSegmentRef.current
        && prompt
      )
        ? prompt === config.environmentDescription.trim()
          ? prompt
          : [
              config.environmentDescription.trim(),
              `Current direction: ${prompt}`,
            ].filter(Boolean).join('\n')
        : config.environmentDescription.trim()
      const commonRequest = {
        camera_trajectory: activeCameraPath?.trajectory,
        trajectory_translation_scale: activeCameraPath ? translationScale : undefined,
        trajectory_rotation_scale: activeCameraPath ? rotationScale : undefined,
        segment_has_camera: Array.from(
          { length: segmentCount },
          () => Boolean(activeCameraPath?.trajectory),
        ),
        user_prompts: activeCameraPath
          ? Array.from({ length: segmentCount }, () => '')
          : Array.from({ length: segmentCount }, () => prompt),
        session_id: sessionIdRef.current,
        reset_session: firstSessionRequestRef.current,
        num_inference_steps: inferenceSteps,
      }
      const requestBody = isOneShot
        ? {
            subject: describedSubject,
            ...commonRequest,
          }
        : structuredSegAr
          ? {
              ...structuredSegAr,
              ...commonRequest,
            }
        : {
            pe_input: {
              session_id: sessionIdRef.current,
              ...(promptEnhancerModel
                ? { prompt_enhancer_model: promptEnhancerModel }
                : {}),
              prompt_enhancer_reasoning_effort:
                promptEnhancerReasoningEffort,
              ...(promptEnhancerSeed === undefined
                ? {}
                : { prompt_enhancer_seed: promptEnhancerSeed }),
              ...(firstSessionRequestRef.current
                ? {
                    perspective: config.perspective,
                    subject_description: describedSubject,
                    environment_description: initialEnvironmentDescription,
                    ...(activeCameraPath
                      ? { camera_controlled: true }
                      : {}),
                  }
                : activeCameraPath
                  ? { camera_controlled: true }
                  : { action: prompt || buildPrompt(prompt) }),
              camera_duration_ms: cameraDurationMs,
            },
            ...commonRequest,
          }

      console.groupCollapsed('[Hy-Nomas] POST /v1/generate')
      console.log('Request URL:', new URL(requestUrl, window.location.href).href)
      console.log('Request body:', requestBody)
      console.groupEnd()

      const response = await fetch(requestUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      })

      const contentType = response.headers.get('content-type') ?? ''
      const rawBody = await response.text()

      console.groupCollapsed(
        `[Hy-Nomas] /v1/generate response: ${response.status} ${response.statusText}`,
      )
      console.log('Response URL:', response.url)
      console.log('Status:', response.status, response.statusText)
      console.log('Content-Type:', contentType || '<missing>')
      console.log('Headers:', Object.fromEntries(response.headers.entries()))
      console.log('Raw response body:', rawBody)
      console.groupEnd()

      if (!contentType.toLowerCase().includes('application/json')) {
        const preview = rawBody.replace(/\s+/g, ' ').trim().slice(0, 300)
        throw new Error(
          `Backend returned HTTP ${response.status} with ${contentType || 'unknown content type'} instead of JSON. Response: ${preview || '<empty body>'}`,
        )
      }

      let payload: GenerateJob | { detail?: string }
      try {
        payload = JSON.parse(rawBody) as GenerateJob | { detail?: string }
      } catch (error) {
        console.error('[Hy-Nomas] Failed to parse JSON response:', error, rawBody)
        throw new Error(
          `Backend returned invalid JSON: ${rawBody.replace(/\s+/g, ' ').trim().slice(0, 300)}`,
        )
      }

      console.log('[Hy-Nomas] Parsed /v1/generate JSON:', payload)
      if (!response.ok) {
        throw new Error('detail' in payload && payload.detail ? payload.detail : `HTTP ${response.status}`)
      }
      let result: GenerateResponse
      if (response.status !== 202) {
        // Backward compatibility with the original synchronous model server.
        result = payload as unknown as GenerateResponse
      } else {
        const submitted = payload as GenerateJob
        let completedResult: GenerateResponse | undefined
        let consecutivePollFailures = 0
        let pollFailureStartedAt: number | undefined
        while (!completedResult) {
          let job: GenerateJob
          try {
            const statusResponse = await fetch(
              `${import.meta.env.BASE_URL}v1/generate/${submitted.id}`,
              { cache: 'no-store' },
            )
            const statusText = await statusResponse.text()
            let statusPayload: GenerateJob | { detail?: string }
            try {
              statusPayload = JSON.parse(statusText) as
                | GenerateJob
                | { detail?: string }
            } catch {
              throw new TransientGenerationPollError(
                `Generation status returned invalid JSON: ${
                  statusText.replace(/\s+/g, ' ').trim().slice(0, 240)
                  || '<empty body>'
                }`,
              )
            }
            if (!statusResponse.ok) {
              const detail =
                'detail' in statusPayload && statusPayload.detail
                  ? statusPayload.detail
                  : `HTTP ${statusResponse.status}`
              if (isTransientGenerationStatus(statusResponse.status)) {
                throw new TransientGenerationPollError(detail)
              }
              throw new Error(detail)
            }
            job = statusPayload as GenerateJob
            consecutivePollFailures = 0
            pollFailureStartedAt = undefined
          } catch (error) {
            const transient =
              error instanceof TypeError
              || error instanceof TransientGenerationPollError
            if (!transient) throw error
            consecutivePollFailures += 1
            pollFailureStartedAt ??= Date.now()
            if (
              Date.now() - pollFailureStartedAt
              > STATUS_POLL_RECOVERY_TIMEOUT_MS
            ) {
              throw new Error(
                'Could not reconnect to the running generation service '
                + 'within 20 minutes. The backend may still retain the '
                + 'result in the Case dashboard.',
                { cause: error },
              )
            }
            setMessages((current) =>
              current.map((message) =>
                message.id === id + 1
                  ? {
                      ...message,
                      content:
                        'Connection interrupted · still reconnecting to the '
                        + `running generation (attempt ${consecutivePollFailures})…`,
                      meta: 'RECONNECTING',
                    }
                  : message,
              ),
            )
            await new Promise((resolve) =>
              window.setTimeout(
                resolve,
                Math.min(
                  5000,
                  STATUS_POLL_RETRY_BASE_MS * 2 ** (
                    consecutivePollFailures - 1
                  ),
                ),
              ),
            )
            continue
          }
          const promptContent = job.pe_prompt ?? job.pe_content_preview
          const finalPrompt = Boolean(job.pe_prompt)
          const jobStatus = job.status ?? 'running'
          setGenerationProgress((current) => {
            if (jobStatus === 'running') {
              return current.phase === 'rendering'
                ? current
                : {
                    phase: 'rendering',
                    startedAt: Date.now(),
                    segmentNumber: 1,
                    totalSegments: 1,
                  }
            }
            if (jobStatus === 'queued') {
              return {
                phase: 'queued',
                queuePosition: job.queue_position,
                segmentNumber: 1,
                totalSegments: 1,
              }
            }
            if (jobStatus === 'enhancing') {
              return { phase: 'preparing', segmentNumber: 1, totalSegments: 1 }
            }
            return current
          })
          if (promptContent) {
            updatePromptTurn(id, {
              k3Output: promptContent,
              k3OutputComplete: finalPrompt,
              structuredPrompt: finalPrompt ? promptContent : undefined,
              requestId: job.pe_request_id ?? submitted.id,
              promptEnhancementMode: job.prompt_enhancement_mode,
              promptEnhancerSeed: job.prompt_enhancer_seed,
            })
          } else if (
            jobStatus === 'enhancing'
            && !job.pe_content_started
            && job.pe_reasoning
          ) {
            updatePromptTurn(id, {
              k3Reasoning: job.pe_reasoning.replace(/\s+/g, ' ').trim(),
              requestId: job.pe_request_id ?? submitted.id,
              promptEnhancementMode: job.prompt_enhancement_mode,
              promptEnhancerSeed: job.prompt_enhancer_seed,
            })
          } else if (job.prompt_enhancement_mode) {
            updatePromptTurn(id, {
              promptEnhancementMode: job.prompt_enhancement_mode,
              promptEnhancerSeed: job.prompt_enhancer_seed,
            })
          }
          setPromptTurnStage(
            id,
            jobStatus === 'enhancing' ? 'simulating' : 'rendering',
          )
          setMessages((current) =>
            current.map((message) =>
              message.id === id + 1
                ? {
                    ...message,
                    thinking:
                      jobStatus === 'enhancing'
                      && !job.pe_content_started
                      && job.pe_reasoning
                        ? job.pe_reasoning.replace(/\s+/g, ' ').trim()
                        : undefined,
                    content:
                      jobStatus === 'enhancing'
                        ? backendConfig?.prompt_enhancement_enabled === false
                          ? 'Preparing the action with the local prompt fallback…'
                          : 'Enhancing the action into a structured segment…'
                        : jobStatus === 'queued'
                          ? job.queue_reason === 'long_session_capacity'
                            ? 'Waiting for long-session capacity · another extended world is using the safe memory budget…'
                            : job.queue_reason === 'long_session_memory_guard'
                              ? 'Waiting for GPU memory · an extended world currently has exclusive safe capacity…'
                            : `Waiting for an available GPU slot${job.queue_position ? ` · ${job.queue_position} job(s) ahead` : ''}…`
                          : 'Generating a new segment-autoregressive world preview…',
                    meta: jobStatus.toUpperCase(),
                  }
                : message,
            ),
          )
          if (jobStatus === 'failed' || jobStatus === 'cancelled') {
            throw new Error(
              `${job.error_code ? `${job.error_code}: ` : ''}${job.error ?? 'Generation failed'}`,
            )
          }
          if (jobStatus === 'succeeded' && job.result) {
            completedResult = job.result
            break
          }
          await new Promise((resolve) =>
            window.setTimeout(resolve, jobStatus === 'enhancing' ? 250 : 1500),
          )
        }
        result = completedResult
      }
      firstSessionRequestRef.current = false
      replacingRolledBackFirstSegmentRef.current = false
      if (
        Number.isFinite(result.generation_seconds)
        && result.generation_seconds > 0
      ) {
        segmentEstimateSecondsRef.current = Math.max(
          8,
          Math.min(
            180,
            segmentEstimateSecondsRef.current * 0.65
              + result.generation_seconds * 0.35,
          ),
        )
      }
      const fps = Math.max(result.fps, 1)
      const segmentFrames = result.segment_frames ?? result.frames
      const startSeconds =
        result.playback_start_seconds
        ?? Math.max(0, (result.frames - segmentFrames) / fps)
      const endSeconds = result.frames / fps
      playingTurnIdRef.current = id
      updatePromptTurn(id, {
        stage: 'playing',
        requestId: result.id,
        frames: segmentFrames,
        fps,
        generationSeconds: result.generation_seconds,
        startSeconds,
        endSeconds,
        playbackProgress: 0,
      })
      setDownloadUrl(`${result.video_url}?request=${encodeURIComponent(result.id)}`)
      setGalleryAdded(false)
      setGalleryError(undefined)
      const streamed = USE_MEDIA_SOURCE_STREAMING
        ? await appendStreamSegment(result)
        : false
      if (!streamed) {
        holdPreviousVideoFrame()
        resetMediaStream()
        pendingResumeTimeRef.current = result.playback_start_seconds ?? 0
        setVideoUrl(`${result.video_url}?request=${encodeURIComponent(result.id)}`)
      }
      setMessages((current) =>
        current.map((message) =>
          message.id === id + 1
            ? {
                ...message,
                content: `Generated ${result.frames} frames at ${result.fps} FPS in ${result.generation_seconds.toFixed(1)} seconds.`,
                meta: 'COMPLETE',
              }
            : message,
        ),
      )
      if (result.prompt_enhanced) {
        completedTurnIdsRef.current.push(id)
        setInteractiveSegmentCount((count) => count + 1)
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      if (message.startsWith('SESSION_EXPIRED:')) {
        onSessionExpiredRef.current(WORLD_SESSION_ENDED_MESSAGE)
        return
      }
      setGenerationError(message)
      updatePromptTurn(id, { stage: 'failed', error: message })
      setMessages((current) =>
        current.map((item) =>
          item.id === id + 1
            ? { ...item, content: `Generation failed: ${message}`, meta: 'FAILED' }
            : item,
        ),
      )
    } finally {
      setIsGenerating(false)
    }
  }

  const rollbackLastSegment = async () => {
    if (isGenerating || interactiveSegmentCount <= 0) return
    setGenerationError(undefined)
    setIsGenerating(true)
    try {
      const sessionId = sessionIdRef.current
      const submitResponse = await fetch(
        `${import.meta.env.BASE_URL}v1/sessions/${encodeURIComponent(sessionId)}/rollback`,
        { method: 'POST' },
      )
      const submitted = (await submitResponse.json()) as
        | RollbackJob
        | { detail?: string }
      if (!submitResponse.ok) {
        throw new Error(
          'detail' in submitted && submitted.detail
            ? submitted.detail
            : `HTTP ${submitResponse.status}`,
        )
      }

      let completed: RollbackJob['result']
      while (!completed) {
        const statusResponse = await fetch(
          `${import.meta.env.BASE_URL}v1/sessions/${encodeURIComponent(sessionId)}/rollback/${(submitted as RollbackJob).id}`,
          { cache: 'no-store' },
        )
        const statusPayload = (await statusResponse.json()) as
          | RollbackJob
          | { detail?: string }
        if (!statusResponse.ok) {
          throw new Error(
            'detail' in statusPayload && statusPayload.detail
              ? statusPayload.detail
              : `HTTP ${statusResponse.status}`,
          )
        }
        const job = statusPayload as RollbackJob
        if (job.status === 'failed' || job.status === 'cancelled') {
          throw new Error(job.error ?? 'Rollback failed')
        }
        if (job.status === 'succeeded' && job.result) {
          completed = job.result
          break
        }
        await new Promise((resolve) => window.setTimeout(resolve, 500))
      }

      resetMediaStream()
      // A restored cumulative MP4 does not contain the prior MSE append
      // history. Keep using cumulative MP4s after rollback so the next
      // generation cannot append a non-zero sequence to an empty SourceBuffer.
      mseFailedRef.current = completed.remaining_segments > 0
      pendingResumeTimeRef.current = 0
      if (completed.video_url) {
        holdPreviousVideoFrame()
        const restoredUrl =
          `${completed.video_url}?rollback=${Date.now()}`
        setVideoUrl(restoredUrl)
        setDownloadUrl(restoredUrl)
      } else {
        setVideoUrl(undefined)
        setDownloadUrl(undefined)
      }
      setGalleryAdded(false)
      setGalleryError(undefined)
      const removedTurnId = completedTurnIdsRef.current.pop()
      const removedTurn = promptTurns.find(
        (turn) => turn.id === removedTurnId,
      )
      if (removedTurnId !== undefined) {
        setPromptTurns((current) =>
          current.filter((turn) => turn.id !== removedTurnId),
        )
        setMessages((current) =>
          current.filter((message) => message.id < removedTurnId),
        )
      }
      if (removedTurn?.cameraPath) {
        setCameraPath(removedTurn.cameraPath)
        setInputMode('camera')
      } else if (removedTurn?.prompt) {
        setDraft(removedTurn.prompt)
        setInputMode('prompt')
      }
      setInteractiveSegmentCount(completed.remaining_segments)
      if (completed.remaining_segments === 0) {
        // The backend deleted the empty session and reset its KV/noise/PixelUMM
        // state. Treat the next user direction as a replacement first segment
        // and resend the complete world context without leaving the Studio.
        firstSessionRequestRef.current = true
        replacingRolledBackFirstSegmentRef.current = true
        completedTurnIdsRef.current = []
        playingTurnIdRef.current = undefined
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setGenerationError(message)
    } finally {
      setIsGenerating(false)
    }
  }

  useEffect(() => {
    if (
      dryRun
      || initialActionStartedRef.current
      || !config.environmentDescription.trim()
      || !backendConfig?.worker_online
    ) {
      return
    }
    initialActionStartedRef.current = true
    void sendPrompt(config.environmentDescription.trim())
  }, [
    backendConfig?.worker_online,
    config.environmentDescription,
    dryRun,
  ])

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    sendPrompt()
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      sendPrompt()
    }
  }

  const copyShareUrl = async (url: string) => {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(url)
      setShareCopied(true)
      return
    }

    const input = document.createElement('textarea')
    input.value = url
    input.style.position = 'fixed'
    input.style.opacity = '0'
    document.body.appendChild(input)
    input.select()
    const copied = document.execCommand?.('copy') ?? false
    input.remove()
    setShareCopied(copied)
  }

  const shareVideo = async () => {
    if (!downloadUrl || isSharing) return
    setIsSharing(true)
    setShareError(undefined)
    setShareCopied(false)
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}v1/shares`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_url: downloadUrl,
          overlay: buildSharedVideoOverlay(promptTurns, theme),
        }),
      })
      const payload = await response.json() as
        | ShareResponse
        | { detail?: string }
      if (!response.ok) {
        throw new Error(
          'detail' in payload && payload.detail
            ? payload.detail
            : `HTTP ${response.status}`,
        )
      }
      const absoluteUrl = new URL(
        (payload as ShareResponse).share_path,
        window.location.href,
      ).toString()
      setShareUrl(absoluteUrl)
      try {
        await copyShareUrl(absoluteUrl)
      } catch {
        // The link remains selectable when clipboard permissions are blocked.
      }
    } catch (error) {
      setShareError(error instanceof Error ? error.message : String(error))
    } finally {
      setIsSharing(false)
    }
  }

  const addToGallery = async () => {
    if (!downloadUrl || isAddingToGallery) return
    setIsAddingToGallery(true)
    setGalleryError(undefined)
    try {
      const lastDirection =
        [...promptTurns].reverse().find((turn) => turn.prompt.trim())?.prompt
        ?? config.environmentDescription.trim()
      const durationSeconds = Math.max(
        videoDuration,
        ...promptTurns.map((turn) => turn.endSeconds ?? 0),
      )
      const response = await fetch(`${import.meta.env.BASE_URL}v1/gallery`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_url: downloadUrl,
          title: config.subjectDescription.trim(),
          perspective: config.perspective,
          subject: config.subjectDescription.trim(),
          attributes: config.environmentDescription.trim(),
          direction: lastDirection,
          prompt_timeline: buildSharedVideoOverlay(promptTurns).turns,
          duration_seconds: durationSeconds,
        }),
      })
      const payload = await response.json() as
        | GalleryResponse
        | { detail?: string }
      if (!response.ok) {
        throw new Error(
          'detail' in payload && payload.detail
            ? payload.detail
            : `HTTP ${response.status}`,
        )
      }
      setGalleryAdded(true)
      onGalleryItemAdded?.((payload as GalleryResponse).item)
    } catch (error) {
      setGalleryError(error instanceof Error ? error.message : String(error))
    } finally {
      setIsAddingToGallery(false)
    }
  }

  const openDownloadDialog = () => {
    if (!downloadUrl) return
    setMoreMenuOpen(false)
    setDownloadMode('overlay')
    setDownloadError(undefined)
    setDownloadDialogOpen(true)
  }

  const triggerDownload = (url: string, filename: string) => {
    const anchor = document.createElement('a')
    anchor.href = new URL(url, window.location.href).toString()
    anchor.download = filename
    anchor.style.display = 'none'
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
  }

  const prepareOverlayDownload = (
    sourceUrl: string,
    turns: PromptTurn[],
  ) => {
    const overlay = buildSharedVideoOverlay(turns, theme)
    const key = `${sourceUrl}\n${JSON.stringify(overlay)}`
    if (overlayDownloadRequestRef.current?.key === key) {
      return overlayDownloadRequestRef.current.request
    }
    setOverlayDownloadStatus('preparing')
    const request = fetch(`${import.meta.env.BASE_URL}v1/downloads`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_url: sourceUrl,
        include_overlay: true,
        overlay,
      }),
    }).then(async (response) => {
      const payload = await response.json() as
        | DownloadResponse
        | { detail?: string }
      if (!response.ok) {
        throw new Error(
          'detail' in payload && payload.detail
            ? payload.detail
            : `HTTP ${response.status}`,
        )
      }
      setOverlayDownloadStatus('ready')
      return payload as DownloadResponse
    })
    overlayDownloadRequestRef.current = { key, request }
    void request.catch(() => {
      setOverlayDownloadStatus('failed')
      if (overlayDownloadRequestRef.current?.request === request) {
        overlayDownloadRequestRef.current = undefined
      }
    })
    return request
  }

  useEffect(() => {
    if (!downloadUrl) {
      overlayDownloadRequestRef.current = undefined
      setOverlayDownloadStatus('idle')
      return
    }
    const overlay = buildSharedVideoOverlay(promptTurns)
    if (!overlay.turns.length) return
    if (import.meta.env.MODE === 'test') return
    // Start the CPU-side MP4 composition as soon as generation completes.
    // In the common case the file is ready before the user opens Download.
    void prepareOverlayDownload(downloadUrl, promptTurns).catch(() => undefined)
  }, [downloadUrl, promptTurns])

  const downloadVideo = async () => {
    if (!downloadUrl || isPreparingDownload) return
    setIsPreparingDownload(true)
    setDownloadError(undefined)
    try {
      if (downloadMode === 'original') {
        const pathname = new URL(downloadUrl, window.location.href).pathname
        const filename = decodeURIComponent(
          pathname.split('/').at(-1) || 'world.mp4',
        )
        triggerDownload(downloadUrl, filename)
      } else {
        const download = await prepareOverlayDownload(
          downloadUrl,
          promptTurns,
        )
        triggerDownload(download.download_path, download.filename)
      }
      setDownloadDialogOpen(false)
    } catch (error) {
      setDownloadError(error instanceof Error ? error.message : String(error))
    } finally {
      setIsPreparingDownload(false)
    }
  }

  const restartPlayback = () => {
    const video = videoRef.current
    if (!videoUrl || !video) return
    video.currentTime = 0
    setVideoCurrentTime(0)
    void video.play().catch(() => {
      // Native controls remain available if autoplay is blocked.
    })
  }

  const exportPePrompts = () => {
    if (interactiveSegmentCount <= 0) return
    const anchor = document.createElement('a')
    anchor.href =
      `${import.meta.env.BASE_URL}v1/sessions/`
      + `${encodeURIComponent(sessionIdRef.current)}/pe-export`
    anchor.download = `session_${sessionIdRef.current}_pe_prompts.json`
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    setMoreMenuOpen(false)
  }

  const toggleFullscreen = async () => {
    const frame = videoFrameRef.current
    const video = videoRef.current as
      | (HTMLVideoElement & { webkitEnterFullscreen?: () => void })
      | null
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen()
      } else if (frame?.requestFullscreen) {
        await frame.requestFullscreen()
      } else {
        video?.webkitEnterFullscreen?.()
      }
    } catch (error) {
      setGenerationError(
        error instanceof Error ? error.message : 'Fullscreen is unavailable',
      )
    }
  }

  return (
    <main className="studio-shell">
      <header className="studio-header">
        <div className="studio-header-left">
          <div className="brand studio-brand">
            <img className="brand-mark" src={`${import.meta.env.BASE_URL}hunyuan-logo.png`} alt="" />
            <span>Hy-Nomas</span>
          </div>
        </div>
        <div className="studio-header-right">
          <ThemeToggle theme={theme} onToggle={onThemeToggle} />
          <ResourceStatus config={backendConfig} dark={theme === 'dark'} />
          <span className="preview-status">
            <i />
            {isGenerating
              ? 'GENERATING'
              : dryRunJob?.status === 'failed'
                ? 'FAILED'
                : videoUrl
                ? `${backendConfig?.rollout_type?.toUpperCase() ?? 'CONNECTED'}`
                : backendConfig
                  ? `${backendConfig.rollout_type.toUpperCase()} READY`
                  : 'READY'}
          </span>
          <div className="more-options" ref={moreMenuRef}>
            <button
              className="icon-button"
              type="button"
              aria-label="More options"
              aria-controls="world-options-menu"
              aria-expanded={moreMenuOpen}
              aria-haspopup="menu"
              onClick={() => setMoreMenuOpen((open) => !open)}
            >
              <MoreHorizontal size={19} />
            </button>
            {moreMenuOpen && (
              <div
                className="more-options-menu"
                id="world-options-menu"
                role="menu"
                aria-label="World options"
              >
                {downloadUrl ? (
                  <button
                    type="button"
                    role="menuitem"
                    onClick={openDownloadDialog}
                  >
                    <Download size={16} />
                    <span>
                      <strong>Download video</strong>
                      <small>Choose overlay or original</small>
                    </span>
                  </button>
                ) : (
                  <button type="button" role="menuitem" disabled>
                    <Download size={16} />
                    <span>
                      <strong>Download video</strong>
                      <small>Available after generation</small>
                    </span>
                  </button>
                )}
                <button
                  type="button"
                  role="menuitem"
                  disabled={!downloadUrl || isSharing}
                  onClick={() => {
                    setMoreMenuOpen(false)
                    void shareVideo()
                  }}
                >
                  <Share2 size={16} />
                  <span>
                    <strong>{isSharing ? 'Creating link…' : 'Share video'}</strong>
                    <small>Anyone with the link can watch</small>
                  </span>
                </button>
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setMoreMenuOpen(false)
                    setGalleryOpen(true)
                  }}
                >
                  <Images size={16} />
                  <span>
                    <strong>View gallery</strong>
                    <small>Browse saved and featured worlds</small>
                    </span>
                </button>
                {backendConfig?.prompt_enhancement_enabled && (
                  <>
                    <label className="more-options-setting">
                      <Bot size={16} />
                      <span><strong>Prompt enhancer</strong></span>
                      <select
                        aria-label="Prompt enhancer model"
                        value={promptEnhancerModel}
                        onChange={(event) => onPromptEnhancerModelChange(event.target.value)}
                      >
                        {(backendConfig.prompt_enhancer_options ?? []).map((option) => (
                          <option key={option.id} value={option.id}>{option.label}</option>
                        ))}
                      </select>
                    </label>
                    <label className="more-options-setting">
                      <Bot size={16} />
                      <span>
                        <strong>Thinking mode</strong>
                        <small>Prompt reasoning depth</small>
                      </span>
                      <select
                        aria-label="Thinking mode"
                        value={promptEnhancerReasoningEffort}
                        onChange={(event) => {
                          onPromptEnhancerReasoningEffortChange(
                            event.target.value as PromptEnhancerReasoningEffort,
                          )
                        }}
                      >
                        {promptEnhancerReasoningEffortOptions.map((option) => (
                          <option key={option.id} value={option.id}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="more-options-setting">
                      <Bot size={16} />
                      <span>
                        <strong>LLM random seed</strong>
                        <small>Blank auto-generates and records one</small>
                      </span>
                      <input
                        aria-label="LLM random seed"
                        type="number"
                        inputMode="numeric"
                        min={promptEnhancerSeedMin}
                        max={promptEnhancerSeedMax}
                        step={1}
                        placeholder="Auto"
                        value={promptEnhancerSeed ?? ''}
                        onChange={(event) => {
                          if (!event.target.value.trim()) {
                            onPromptEnhancerSeedChange(undefined)
                            return
                          }
                          const seed = Number(event.target.value)
                          if (
                            Number.isSafeInteger(seed)
                            && seed >= promptEnhancerSeedMin
                            && seed <= promptEnhancerSeedMax
                          ) {
                            onPromptEnhancerSeedChange(seed)
                          }
                        }}
                      />
                    </label>
                  </>
                )}
                <label className="more-options-setting">
                  <Clapperboard size={16} />
                  <span>
                    <strong>Inference steps</strong>
                    <small>Denoising quality and speed</small>
                  </span>
                  <select
                    aria-label="Inference steps"
                    value={inferenceSteps}
                    onChange={(event) => {
                      onInferenceStepsChange(
                        Number(event.target.value) as InferenceSteps,
                      )
                    }}
                  >
                    {inferenceStepsOptions.map((steps) => (
                      <option key={steps} value={steps}>
                        {steps}
                        {steps === 4
                          ? ' · Fast'
                          : steps === 6
                            ? ' · Balanced'
                            : ' · Quality'}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  role="menuitem"
                  disabled={interactiveSegmentCount <= 0}
                  onClick={exportPePrompts}
                >
                  <Download size={16} />
                  <span>
                    <strong>Export PE prompts</strong>
                    <small>Original, enhanced, model, and camera JSON</small>
                  </span>
                </button>
                {ROLLBACK_CONTROLS_ENABLED && (
                  <button
                    type="button"
                    role="menuitem"
                    disabled={isGenerating || interactiveSegmentCount <= 0}
                    onClick={() => {
                      setMoreMenuOpen(false)
                      void rollbackLastSegment()
                    }}
                  >
                    <RotateCcw size={16} />
                    <span>
                      <strong>Redo last segment</strong>
                      <small>Restore its prompt, edit it, then generate again</small>
                    </span>
                  </button>
                )}
              </div>
            )}
          </div>
          <button className="new-world-button" type="button" onClick={onReconfigure}>
            <Plus size={15} />
            New world
          </button>
        </div>
      </header>

      <div className="studio-layout">
        <section className="video-pane" aria-label="Generated world">
          <div className="video-toolbar">
            <div className="video-label">
              <Clapperboard size={15} />
              <strong>World preview</strong>
              <span>
                {formatPlaybackTime(videoCurrentTime)}
                {' / '}
                {formatPlaybackTime(videoDuration)}
              </span>
            </div>
            <button
              className="expand-button"
              type="button"
              aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
              title={isFullscreen ? 'Exit fullscreen' : 'Play fullscreen'}
              disabled={!videoUrl}
              onClick={() => void toggleFullscreen()}
            >
              <Expand size={18} />
            </button>
          </div>

          <div className="video-frame" ref={videoFrameRef}>
            <video
              ref={videoRef}
              aria-label="World model output"
              poster={`${import.meta.env.BASE_URL}world-placeholder.svg`}
              src={videoUrl}
              controls
              muted
              playsInline
              onLoadedMetadata={() => {
                const video = videoRef.current
                if (!video) return
                setVideoDuration(
                  Number.isFinite(video.duration) ? video.duration : 0,
                )
                setVideoResolution(
                  video.videoWidth && video.videoHeight
                    ? `${video.videoWidth}×${video.videoHeight}`
                    : '',
                )
                if (pendingResumeTimeRef.current === undefined) return
                const resumeAt = pendingResumeTimeRef.current
                const targetTime = Math.min(
                  resumeAt,
                  Math.max(video.duration - 0.01, 0),
                )
                if (Math.abs(video.currentTime - targetTime) > 0.02) {
                  video.currentTime = targetTime
                } else {
                  pendingResumeTimeRef.current = undefined
                  void video.play().catch(() => {
                    // Browser autoplay policy may require an explicit click.
                  })
                }
              }}
              onCanPlay={(event) => {
                if (pendingResumeTimeRef.current === undefined) {
                  releasePreviousVideoFrame()
                  void event.currentTarget.play().catch(() => {
                    // Browser autoplay policy may require an explicit click.
                  })
                }
              }}
              onTimeUpdate={(event) => {
                if (transitionFrameActiveRef.current) return
                const activeTurnId = playingTurnIdRef.current
                const currentTime = event.currentTarget.currentTime
                setVideoCurrentTime(currentTime)
                if (activeTurnId === undefined) return
                updatePromptTurn(activeTurnId, (activeTurn) => {
                  const start = activeTurn.startSeconds ?? 0
                  const end = activeTurn.endSeconds ?? start
                  const duration = Math.max(end - start, 1 / Math.max(activeTurn.fps ?? 24, 1))
                  const progress = Math.max(
                    0,
                    Math.min(100, ((currentTime - start) / duration) * 100),
                  )
                  return { playbackProgress: progress }
                })
              }}
              onSeeked={(event) => {
                const video = event.currentTarget
                setVideoCurrentTime(video.currentTime)
                if (pendingResumeTimeRef.current !== undefined) {
                  pendingResumeTimeRef.current = undefined
                  void video.play().catch(() => {
                    // Browser autoplay policy may require an explicit click.
                  })
                  if (video.readyState >= 3) {
                    releasePreviousVideoFrame()
                  }
                }
              }}
              onPlaying={() => {
                if (pendingResumeTimeRef.current === undefined) {
                  releasePreviousVideoFrame()
                }
              }}
              onEnded={() => {
                const video = videoRef.current
                setVideoCurrentTime(video?.currentTime ?? 0)
                if (video) rememberLastVideoFrame(video)
                const activeTurnId = playingTurnIdRef.current
                if (activeTurnId === undefined) return
                updatePromptTurn(activeTurnId, {
                  stage: 'complete',
                  playbackProgress: 100,
                })
                playingTurnIdRef.current = undefined
              }}
            />
            {transitionFrame && (
              <img
                className={[
                  'video-transition-frame',
                  transitionFrameVisible ? 'visible' : '',
                ].filter(Boolean).join(' ')}
                src={transitionFrame}
                alt=""
                aria-hidden="true"
              />
            )}
            {videoUrl && playbackTurn && (
              playbackTurn.cameraPath ? (
                <CameraPlaybackOverlay
                  path={playbackTurn.cameraPath}
                  videoRef={videoRef}
                  segmentStartSeconds={playbackTurn.startSeconds ?? 0}
                  segmentDurationSeconds={Math.max(
                    0,
                    (playbackTurn.endSeconds ?? 0)
                      - (playbackTurn.startSeconds ?? 0),
                  )}
                  visible={userPromptVisible}
                  onVisibilityToggle={() => {
                    setUserPromptVisible((visible) => !visible)
                  }}
                />
              ) : (
                <DirectionHud
                  direction={playbackTurn.prompt}
                  visible={userPromptVisible}
                  onVisibilityToggle={() => {
                    setUserPromptVisible((visible) => !visible)
                  }}
                />
              )
            )}
            {videoUrl && userPromptVisible && (
              <DemoOverlayTimeline turns={promptTurns} currentTime={videoCurrentTime} />
            )}
            {isGenerating && (
              <div className="video-generation-overlay" aria-live="polite">
                <div className="video-generation-progress-copy">
                  <span>{progressLabel}</span>
                  <span>{progressEta}</span>
                </div>
                <div
                  className={[
                    'video-generation-progress',
                    generationProgress.phase !== 'rendering' ? 'indeterminate' : '',
                  ].filter(Boolean).join(' ')}
                  role="progressbar"
                  aria-label="Current segment generation progress"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={
                    generationProgress.phase === 'rendering'
                      ? Math.round(segmentProgressPercent)
                      : undefined
                  }
                >
                  <i style={{ width: `${segmentProgressPercent}%` }} />
                </div>
                <div className="video-generation-message">
                  <span className="video-generation-spinner" />
                  <strong>SIMULATING WHAT HAPPENS NEXT</strong>
                </div>
              </div>
            )}
            <div className="video-corner corner-top-left" />
            <div className="video-corner corner-top-right" />
            <div className="video-corner corner-bottom-left" />
            <div className="video-corner corner-bottom-right" />
          </div>

          <footer className="video-footer">
            <div className="world-specs">
              <span>
                <small>Perspective</small>
                <strong>{perspectiveLabel(config.perspective)}</strong>
              </span>
              <span>
                <small>Format</small>
                <strong>16:9</strong>
              </span>
              <span>
                <small>Quality</small>
                <strong>{videoResolution || '—'}</strong>
              </span>
              <span className="world-spec-model">
                <small>Model</small>
                <strong title={inferenceModelLabel}>
                  {inferenceModelLabel}
                </strong>
              </span>
            </div>
            <div className="video-actions">
              <button
                type="button"
                onClick={() => void shareVideo()}
                disabled={!downloadUrl || isSharing}
              >
                <Share2 size={15} />
                {isSharing ? 'Creating…' : 'Share'}
              </button>
              <button
                type="button"
                onClick={openDownloadDialog}
                disabled={!downloadUrl}
              >
                <Download size={15} />
                Download
              </button>
              <button
                type="button"
                onClick={() => void addToGallery()}
                disabled={!downloadUrl || isAddingToGallery || galleryAdded}
                title={galleryError}
              >
                {galleryAdded ? <Check size={15} /> : <Plus size={15} />}
                {galleryAdded
                  ? 'Added'
                  : isAddingToGallery
                    ? 'Adding…'
                    : galleryError
                      ? 'Try again'
                      : 'Add to gallery'}
              </button>
              <button
                type="button"
                onClick={restartPlayback}
                disabled={!videoUrl}
                aria-label="Replay from start"
              >
                <RotateCcw size={15} />
                Replay
              </button>
            </div>
          </footer>
        </section>

        <aside className="agent-pane" aria-label="World simulator">
          <header className="agent-header">
            <div>
              <span className="agent-avatar">
                <Bot size={16} />
              </span>
              <div>
                <strong>World Simulator</strong>
              </div>
            </div>
          </header>

          <div className="world-context">
            <span>{dryRun ? 'DRY RUN SAMPLE' : isOneShot ? 'PROMPT FILE' : 'SUBJECT'}</span>
            <p>
              {dryRun
                ? dryRunJob?.sample_id ?? 'Selecting…'
                : describedSubject}
            </p>
          </div>

          <div className="message-list" aria-live="polite">
            {dryRun ? messages.map((message) => (
              <article className={`message ${message.role}`} key={message.id}>
                <div className="message-meta">
                  <span>{message.role === 'agent' ? <Bot size={12} /> : 'YOU'}</span>
                  <time>{message.meta}</time>
                </div>
                <p>{message.content}</p>
              </article>
            )) : <PromptTurnTimeline turns={promptTurns} />}
          </div>

          {!dryRun && <div className={`composer-area${inputMode === 'camera' ? ' camera-mode' : ''}`}>
                <div className="direction-mode" aria-label="Generation input type">
                  <button
                    type="button"
                    className={inputMode === 'prompt' ? 'active' : ''}
                    aria-pressed={inputMode === 'prompt'}
                    onClick={() => setInputMode('prompt')}
                    disabled={isGenerating}
                  >
                    <Bot size={12} />
                    Prompt
                  </button>
                  <span>OR</span>
                  <button
                    type="button"
                    className={inputMode === 'camera' ? 'active' : ''}
                    aria-pressed={inputMode === 'camera'}
                    onClick={() => setInputMode('camera')}
                    disabled={isGenerating}
                  >
                    <Camera size={12} />
                    Camera path
                  </button>
                </div>

                {inputMode === 'prompt' ? (
                  <form className="composer" onSubmit={submit}>
                      <textarea
                        aria-label="Direct the world"
                        value={draft}
                        onChange={(event) => setDraft(event.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Describe what should happen next…"
                        rows={3}
                        disabled={isGenerating}
                      />
                      <div className="composer-footer">
                        <span>SHIFT + ENTER FOR NEW LINE</span>
                        <button
                          type="submit"
                          aria-label="Send prompt"
                          disabled={!draft.trim() || isGenerating}
                        >
                          <ArrowUp size={17} strokeWidth={2.5} />
                        </button>
                      </div>
                  </form>
                ) : (
                  <div className="camera-only-composer">
                    <CameraPathDrawer
                      ref={cameraDrawerRef}
                      embedded
                      selected={cameraPath}
                      disabled={isGenerating}
                      theme={theme}
                      translationScale={translationScale}
                      rotationScale={rotationScale}
                      onTranslationScaleChange={(translation) => {
                        setCameraScaleOverrides((current) => ({ ...current, translation }))
                      }}
                      onRotationScaleChange={(rotation) => {
                        setCameraScaleOverrides((current) => ({ ...current, rotation }))
                      }}
                      onApply={setCameraPath}
                      onClear={() => setCameraPath(undefined)}
                      onRecordingStateChange={(recording) => {
                        setIsCameraRecording(recording)
                        if (recording) setCameraPath(undefined)
                      }}
                    />
                    {cameraPath && (
                      <div className="camera-selected-summary">
                        <span>
                          Ready to generate
                          {cameraPathDurationMs(cameraPath)
                            ? ` · ${(cameraPathDurationMs(cameraPath)! / 1000).toFixed(2)}s`
                            : ''}
                          {cameraPath.frames?.length
                            ? ` · ${cameraPath.frames.length} frames @ ${cameraPath.sampleRate ?? 24} FPS`
                            : ''}
                        </span>
                        <button
                          type="button"
                          aria-label="Remove camera path"
                          onClick={() => setCameraPath(undefined)}
                          disabled={isGenerating}
                        >
                          <X size={12} />
                        </button>
                      </div>
                    )}
                    <button
                      className="camera-generate-button"
                      type="button"
                      aria-label="Generate with camera path"
                      disabled={(!cameraPath && !isCameraRecording) || isGenerating}
                      onClick={() => {
                        const finalizedPath = cameraDrawerRef.current?.finalizeRecording() ?? cameraPath
                        if (!finalizedPath) return
                        void sendPrompt('', finalizedPath)
                        cameraDrawerRef.current?.resetRecording()
                      }}
                    >
                      Generate with camera path
                      <ArrowUp size={15} />
                    </button>
                  </div>
                )}
                <p className="preview-note">
                  {generationError
                    ? `Backend error: ${generationError}`
                    : isGenerating
                      ? 'The selected input is locked while the GPU renders.'
                      : inputMode === 'prompt'
                        ? 'Enter a prompt, or switch to Camera path.'
                        : 'Select one trajectory; no prompt is required.'}
                </p>
          </div>}
        </aside>
      </div>
      {(shareUrl || shareError) && (
        <div
          className="share-dialog-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setShareUrl(undefined)
              setShareError(undefined)
            }
          }}
        >
          <section
            className="share-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="share-dialog-title"
          >
            <header>
              <span className="share-dialog-icon">
                <Share2 size={19} />
              </span>
              <div>
                <small>PUBLIC VIDEO LINK</small>
                <h2 id="share-dialog-title">
                  {shareError ? 'Could not create link' : 'Ready to share'}
                </h2>
              </div>
              <button
                type="button"
                aria-label="Close share dialog"
                onClick={() => {
                  setShareUrl(undefined)
                  setShareError(undefined)
                }}
              >
                <X size={17} />
              </button>
            </header>
            {shareError ? (
              <p className="share-dialog-error">{shareError}</p>
            ) : (
              <>
                <p>
                  Anyone with this link can watch the generated video without
                  signing in.
                </p>
                <div className="share-link-row">
                  <input
                    aria-label="Public share link"
                    value={shareUrl}
                    readOnly
                    onFocus={(event) => event.currentTarget.select()}
                  />
                  <button
                    type="button"
                    onClick={() => {
                      if (shareUrl) void copyShareUrl(shareUrl)
                    }}
                  >
                    {shareCopied ? <Check size={16} /> : <Copy size={16} />}
                    {shareCopied ? 'Copied' : 'Copy'}
                  </button>
                </div>
              </>
            )}
          </section>
        </div>
      )}
      {downloadDialogOpen && (
        <div
          className="share-dialog-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (
              event.target === event.currentTarget
              && !isPreparingDownload
            ) {
              setDownloadDialogOpen(false)
            }
          }}
        >
          <section
            className="share-dialog download-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="download-dialog-title"
          >
            <header>
              <span className="share-dialog-icon">
                <Download size={19} />
              </span>
              <div>
                <small>DOWNLOAD VIDEO</small>
                <h2 id="download-dialog-title">Choose a version</h2>
              </div>
              <button
                type="button"
                aria-label="Close download dialog"
                disabled={isPreparingDownload}
                onClick={() => setDownloadDialogOpen(false)}
              >
                <X size={17} />
              </button>
            </header>
            <p>
              The overlay version includes the synchronized prompt and camera
              direction controls.
            </p>
            <div className="download-options" role="radiogroup" aria-label="Video version">
              <label className={downloadMode === 'overlay' ? 'selected' : ''}>
                <input
                  type="radio"
                  name="download-mode"
                  value="overlay"
                  checked={downloadMode === 'overlay'}
                  onChange={() => setDownloadMode('overlay')}
                />
                <span>
                  <strong>With overlay</strong>
                  <small>
                    Prompt and camera direction keys · Recommended
                    {overlayDownloadStatus === 'preparing'
                      ? ' · Preparing in background…'
                      : overlayDownloadStatus === 'ready'
                        ? ' · Ready'
                        : ''}
                  </small>
                </span>
                {downloadMode === 'overlay' && <Check size={17} />}
              </label>
              <label className={downloadMode === 'original' ? 'selected' : ''}>
                <input
                  type="radio"
                  name="download-mode"
                  value="original"
                  checked={downloadMode === 'original'}
                  onChange={() => setDownloadMode('original')}
                />
                <span>
                  <strong>Original video</strong>
                  <small>Clean MP4 without prompt or camera overlays</small>
                </span>
                {downloadMode === 'original' && <Check size={17} />}
              </label>
            </div>
            {downloadError && (
              <p className="share-dialog-error">{downloadError}</p>
            )}
            <div className="download-dialog-actions">
              <button
                type="button"
                className="secondary-button"
                disabled={isPreparingDownload}
                onClick={() => setDownloadDialogOpen(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="primary-button"
                disabled={isPreparingDownload}
                onClick={() => void downloadVideo()}
              >
                <Download size={16} />
                {isPreparingDownload
                  ? 'Finishing download…'
                  : downloadMode === 'overlay'
                    ? 'Download with overlay'
                    : 'Download original'}
              </button>
            </div>
          </section>
        </div>
      )}
      {galleryOpen && (
        <div
          className="studio-gallery-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setGalleryOpen(false)
            }
          }}
        >
          <section
            className="studio-gallery-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="studio-gallery-title"
          >
            <header>
              <div>
                <small>WORLD GALLERY</small>
                <h2 id="studio-gallery-title">Saved and featured worlds</h2>
              </div>
              <button
                type="button"
                aria-label="Close world gallery"
                onClick={() => setGalleryOpen(false)}
              >
                <X size={18} />
              </button>
            </header>
            <WorldGallery
              savedSamples={backendConfig?.gallery_items}
              hiddenSampleIds={backendConfig?.gallery_hidden_ids}
              viewOnly
            />
          </section>
        </div>
      )}
    </main>
  )
}

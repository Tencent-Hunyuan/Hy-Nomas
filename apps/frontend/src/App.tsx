import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { SetupScreen } from './components/SetupScreen'
import { WorldStudio } from './components/WorldStudio'
import type {
  ActiveAllocationsResponse,
  BackendConfig,
  DispatchSession,
  GalleryItem,
  GalleryItemRemovedDetail,
  InferenceSteps,
  PromptEnhancerModel,
  PromptEnhancerReasoningEffort,
  PromptEnhancerReasoningEffortOption,
  PromptEnhancerSeed,
  WorldConfig,
} from './types'
import { GALLERY_ITEM_REMOVED_EVENT } from './types'
import { normalizePerspective } from './perspective'

const browserClientId =
  globalThis.crypto?.randomUUID?.()
  ?? `nomas-browser-${Date.now()}-${Math.random().toString(36).slice(2)}`

function createDispatcherPageId() {
  return (
    globalThis.crypto?.randomUUID?.()
    ?? `nomas-page-${Date.now()}-${Math.random().toString(36).slice(2)}`
  )
}

const MULTI_HOST = import.meta.env.VITE_MULTI_HOST === 'true'
const STUDIO_PREVIEW =
  import.meta.env.DEV
  && new URLSearchParams(window.location.search).get('studio-preview') === '1'
const THEME_STORAGE_KEY = 'nomas-world-engine-theme'
const PROMPT_ENHANCER_STORAGE_KEY = 'nomas-prompt-enhancer-model'
const PROMPT_ENHANCER_REASONING_EFFORT_STORAGE_KEY =
  'nomas-prompt-enhancer-reasoning-effort'
const PROMPT_ENHANCER_SEED_STORAGE_KEY = 'nomas-prompt-enhancer-seed'
const PERSPECTIVE_STORAGE_KEY = 'nomas-world-perspective'
const INFERENCE_STEPS_STORAGE_KEY = 'nomas-num-inference-steps'
const DEFAULT_PROMPT_ENHANCER_MODEL: PromptEnhancerModel = 'Nomas-PE'
const DEFAULT_INFERENCE_STEPS: InferenceSteps = 8
const DEFAULT_INFERENCE_STEPS_OPTIONS: InferenceSteps[] = [4, 6, 8]
const DEFAULT_PROMPT_ENHANCER_REASONING_EFFORT:
  PromptEnhancerReasoningEffort = 'none'
const DEFAULT_PROMPT_ENHANCER_SEED_MIN = 0
const DEFAULT_PROMPT_ENHANCER_SEED_MAX = 2 ** 31 - 1
const DEFAULT_PROMPT_ENHANCER_REASONING_EFFORT_OPTIONS:
  PromptEnhancerReasoningEffortOption[] = [
    { id: 'none', label: 'None', default: true },
    { id: 'low', label: 'Low' },
    { id: 'medium', label: 'Medium' },
    { id: 'high', label: 'High' },
    { id: 'xhigh', label: 'Extra high' },
  ]
const ALLOCATIONS_FETCH_MAX_ATTEMPTS = 5
const ALLOCATIONS_FETCH_RETRY_BASE_MS = 300
const ALLOCATIONS_RETRYABLE_STATUSES = new Set([
  408,
  425,
  429,
  500,
  502,
  503,
  504,
])

const STUDIO_PREVIEW_CONFIG: BackendConfig = {
  rollout_type: 'seg_ar',
  prompt_source: 'prompt',
  num_frames: 80,
  num_frames_source: 'request_or_config',
  fps: 24,
  worker_online: true,
  worker_status: 'ready',
  model_name: 'Stage 4 SFT',
}

interface DispatcherHealth {
  hosts?: Array<{
    online?: boolean
    available?: boolean
  }>
  active_users?: number
  waitlist_size?: number
}

function initialTheme(): 'dark' | 'light' {
  try {
    return window.localStorage.getItem(THEME_STORAGE_KEY) === 'light'
      ? 'light'
      : 'dark'
  } catch {
    return 'dark'
  }
}

function initialWorldConfig(): WorldConfig {
  let perspective = normalizePerspective(undefined)
  try {
    perspective = normalizePerspective(
      window.localStorage.getItem(PERSPECTIVE_STORAGE_KEY),
    )
    window.localStorage.setItem(PERSPECTIVE_STORAGE_KEY, perspective)
  } catch {
    // Storage can be unavailable in restricted browsers.
  }
  return {
    perspective,
    subjectDescription: '',
    environmentDescription: '',
  }
}

function isInferenceSteps(value: unknown): value is InferenceSteps {
  return typeof value === 'number' && Number.isInteger(value) && value > 0
}

function initialInferenceSteps(): InferenceSteps | undefined {
  try {
    const raw = window.localStorage.getItem(INFERENCE_STEPS_STORAGE_KEY)
    if (raw === null) return undefined
    const stored = Number(raw)
    return isInferenceSteps(stored) ? stored : undefined
  } catch {
    return undefined
  }
}

function inferenceStepsOptions(
  backendConfig: BackendConfig | undefined,
): InferenceSteps[] {
  const normalized = (backendConfig?.num_inference_steps_options ?? [])
    .filter(isInferenceSteps)
    .filter((value, index, values) => values.indexOf(value) === index)
  return normalized.length ? normalized : DEFAULT_INFERENCE_STEPS_OPTIONS
}

function defaultInferenceSteps(
  backendConfig: BackendConfig | undefined,
  options: InferenceSteps[],
): InferenceSteps {
  for (const candidate of [
    backendConfig?.default_num_inference_steps,
    backendConfig?.num_inference_steps,
    DEFAULT_INFERENCE_STEPS,
    options[0],
  ]) {
    if (
      isInferenceSteps(candidate)
      && options.includes(candidate)
    ) {
      return candidate
    }
  }
  return DEFAULT_INFERENCE_STEPS
}

function isPromptEnhancerReasoningEffort(
  value: unknown,
): value is PromptEnhancerReasoningEffort {
  return value === 'none'
    || value === 'low'
    || value === 'medium'
    || value === 'high'
    || value === 'xhigh'
}

function initialPromptEnhancerReasoningEffort():
  PromptEnhancerReasoningEffort | undefined {
  try {
    const stored = window.localStorage.getItem(
      PROMPT_ENHANCER_REASONING_EFFORT_STORAGE_KEY,
    )
    return isPromptEnhancerReasoningEffort(stored) ? stored : undefined
  } catch {
    return undefined
  }
}

function promptEnhancerSeedRange(
  backendConfig: BackendConfig | undefined,
): { min: number; max: number } {
  const min = Number.isSafeInteger(backendConfig?.prompt_enhancer_seed_min)
    ? Number(backendConfig?.prompt_enhancer_seed_min)
    : DEFAULT_PROMPT_ENHANCER_SEED_MIN
  const max = Number.isSafeInteger(backendConfig?.prompt_enhancer_seed_max)
    ? Number(backendConfig?.prompt_enhancer_seed_max)
    : DEFAULT_PROMPT_ENHANCER_SEED_MAX
  return min <= max
    ? { min, max }
    : {
        min: DEFAULT_PROMPT_ENHANCER_SEED_MIN,
        max: DEFAULT_PROMPT_ENHANCER_SEED_MAX,
      }
}

function initialPromptEnhancerSeed(): PromptEnhancerSeed | undefined {
  try {
    const raw = window.localStorage.getItem(PROMPT_ENHANCER_SEED_STORAGE_KEY)
    if (raw === null || !raw.trim()) return undefined
    const seed = Number(raw)
    return Number.isSafeInteger(seed)
      && seed >= DEFAULT_PROMPT_ENHANCER_SEED_MIN
      && seed <= DEFAULT_PROMPT_ENHANCER_SEED_MAX
      ? seed
      : undefined
  } catch {
    return undefined
  }
}

function normalizeReasoningEffortOptions(
  options: Array<
    PromptEnhancerReasoningEffort | PromptEnhancerReasoningEffortOption
  > | undefined,
): PromptEnhancerReasoningEffortOption[] {
  if (!options?.length) return []
  const normalized: PromptEnhancerReasoningEffortOption[] = []
  for (const option of options) {
    const id = typeof option === 'string' ? option : option.id
    if (
      !isPromptEnhancerReasoningEffort(id)
      || normalized.some((candidate) => candidate.id === id)
    ) {
      continue
    }
    normalized.push({
      id,
      label: typeof option === 'string'
        ? `${option.charAt(0).toUpperCase()}${option.slice(1)}`
        : option.label,
      default: typeof option === 'string' ? undefined : option.default,
    })
  }
  return normalized
}

function reasoningEffortOptionsForModel(
  backendConfig: BackendConfig | undefined,
  model: PromptEnhancerModel,
): PromptEnhancerReasoningEffortOption[] {
  const modelOption = backendConfig?.prompt_enhancer_options?.find(
    (option) => option.id === model,
  )
  const configured = modelOption?.reasoning_effort_options
    ?? (
      Array.isArray(backendConfig?.prompt_enhancer_reasoning_effort_options)
        ? backendConfig.prompt_enhancer_reasoning_effort_options
        : backendConfig?.prompt_enhancer_reasoning_effort_options?.[model]
    )
  const normalized = normalizeReasoningEffortOptions(configured)
  return normalized.length
    ? normalized
    : DEFAULT_PROMPT_ENHANCER_REASONING_EFFORT_OPTIONS
}

function defaultReasoningEffortForModel(
  backendConfig: BackendConfig | undefined,
  model: PromptEnhancerModel,
  options: PromptEnhancerReasoningEffortOption[],
): PromptEnhancerReasoningEffort {
  const modelOption = backendConfig?.prompt_enhancer_options?.find(
    (option) => option.id === model,
  )
  const configuredDefault =
    modelOption?.default_reasoning_effort
    ?? (
      typeof backendConfig?.default_prompt_enhancer_reasoning_effort === 'string'
        ? backendConfig.default_prompt_enhancer_reasoning_effort
        : backendConfig?.default_prompt_enhancer_reasoning_effort?.[model]
    )
  if (
    isPromptEnhancerReasoningEffort(configuredDefault)
    && options.some((option) => option.id === configuredDefault)
  ) {
    return configuredDefault
  }
  return options.find((option) => option.default)?.id
    ?? options.find(
      (option) => option.id === DEFAULT_PROMPT_ENHANCER_REASONING_EFFORT,
    )?.id
    ?? options[0]?.id
    ?? DEFAULT_PROMPT_ENHANCER_REASONING_EFFORT
}

async function parseJsonResponse<T>(response: Response, endpoint: string): Promise<T> {
  const contentType = response.headers.get('content-type') ?? ''
  const rawBody = await response.text()
  try {
    return JSON.parse(rawBody) as T
  } catch {
    const preview = rawBody.trim().slice(0, 240) || '<empty response>'
    throw new Error(
      `${endpoint} returned invalid JSON: HTTP ${response.status}, `
      + `Content-Type ${contentType || '<missing>'}, body: ${preview}`,
    )
  }
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

function waitForRetry(delayMs: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException('The operation was aborted.', 'AbortError'))
      return
    }

    const timer = window.setTimeout(() => {
      signal.removeEventListener('abort', abort)
      resolve()
    }, delayMs)
    const abort = () => {
      window.clearTimeout(timer)
      reject(new DOMException('The operation was aborted.', 'AbortError'))
    }
    signal.addEventListener('abort', abort, { once: true })
  })
}

interface AppProps {
  multiHost?: boolean
}

function App({ multiHost = MULTI_HOST }: AppProps = {}) {
  const pageClosingRef = useRef(false)
  const dispatcherPageIdRef = useRef(createDispatcherPageId())
  const logoutAfterWorldReleaseRef = useRef(false)
  const allocationsRequestRef = useRef<AbortController | undefined>(undefined)
  const mountedRef = useRef(true)
  const [theme, setTheme] = useState<'dark' | 'light'>(initialTheme)
  const [storedPromptEnhancerModel, setPromptEnhancerModel] = useState(() => {
    try {
      return window.localStorage.getItem(PROMPT_ENHANCER_STORAGE_KEY)?.trim()
        || undefined
    } catch {
      return undefined
    }
  })
  const [inferenceSteps, setInferenceSteps] =
    useState<InferenceSteps | undefined>(initialInferenceSteps)
  const [
    promptEnhancerReasoningEffort,
    setPromptEnhancerReasoningEffort,
  ] = useState<PromptEnhancerReasoningEffort | undefined>(
    initialPromptEnhancerReasoningEffort,
  )
  const [promptEnhancerSeed, setPromptEnhancerSeed] =
    useState<PromptEnhancerSeed | undefined>(initialPromptEnhancerSeed)
  const [dispatchSession, setDispatchSession] = useState<DispatchSession>()
  const [loginName, setLoginName] = useState('')
  const [loginError, setLoginError] = useState<string>()
  const [loggingIn, setLoggingIn] = useState(false)
  const [showAllocations, setShowAllocations] = useState(false)
  const [allocations, setAllocations] = useState<ActiveAllocationsResponse>()
  const [allocationsError, setAllocationsError] = useState<string>()
  const [allocationsLoading, setAllocationsLoading] = useState(false)
  const [screen, setScreen] = useState<'setup' | 'studio'>(
    STUDIO_PREVIEW ? 'studio' : 'setup',
  )
  const [config, setConfig] = useState<WorldConfig>(() => (
    STUDIO_PREVIEW
      ? {
          perspective: 'third-person follow',
          subjectDescription: 'A cinematic third-person world',
          environmentDescription: 'Frontend-only Studio preview',
        }
      : initialWorldConfig()
  ))
  const [backendConfig, setBackendConfig] = useState<BackendConfig | undefined>(
    STUDIO_PREVIEW ? STUDIO_PREVIEW_CONFIG : undefined,
  )
  const [backendError, setBackendError] = useState<string>()
  const [dispatcherHealth, setDispatcherHealth] =
    useState<DispatcherHealth>()
  const [dryRun, setDryRun] = useState(false)
  const [sessionNotice, setSessionNotice] = useState<string>()

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      allocationsRequestRef.current?.abort()
    }
  }, [])

  const promptEnhancerModel = useMemo(() => {
    const options = backendConfig?.prompt_enhancer_options
    if (!options) return storedPromptEnhancerModel ?? DEFAULT_PROMPT_ENHANCER_MODEL
    if (options.some((option) => option.id === storedPromptEnhancerModel)) {
      return storedPromptEnhancerModel!
    }
    return options.find(
      (option) => option.id === backendConfig?.default_prompt_enhancer_model,
    )?.id
      ?? options.find((option) => option.default)?.id
      ?? options[0]?.id
      ?? ''
  }, [backendConfig, storedPromptEnhancerModel])

  const selectPromptEnhancerModel = useCallback((model: PromptEnhancerModel) => {
    setPromptEnhancerModel(model)
    try {
      window.localStorage.setItem(PROMPT_ENHANCER_STORAGE_KEY, model)
    } catch {
      // Storage can be unavailable in restricted browsers.
    }
  }, [])

  useEffect(() => {
    if (backendConfig?.prompt_enhancer_options?.length
      && storedPromptEnhancerModel !== promptEnhancerModel) {
      selectPromptEnhancerModel(promptEnhancerModel)
    }
  }, [backendConfig, storedPromptEnhancerModel, promptEnhancerModel, selectPromptEnhancerModel])

  const promptEnhancerReasoningEffortOptions = useMemo(
    () => reasoningEffortOptionsForModel(
      backendConfig,
      promptEnhancerModel,
    ),
    [backendConfig, promptEnhancerModel],
  )
  const availableInferenceSteps = useMemo(
    () => inferenceStepsOptions(backendConfig),
    [backendConfig],
  )
  const promptEnhancerSeedRangeValue = useMemo(
    () => promptEnhancerSeedRange(backendConfig),
    [backendConfig],
  )
  const resolvedInferenceSteps = useMemo(() => (
    inferenceSteps !== undefined && availableInferenceSteps.includes(inferenceSteps)
      ? inferenceSteps
      : defaultInferenceSteps(backendConfig, availableInferenceSteps)
  ), [availableInferenceSteps, backendConfig, inferenceSteps])
  const resolvedPromptEnhancerReasoningEffort = useMemo(() => {
    if (
      promptEnhancerReasoningEffortOptions.some(
        (option) => option.id === promptEnhancerReasoningEffort,
      )
    ) {
      return promptEnhancerReasoningEffort
        ?? DEFAULT_PROMPT_ENHANCER_REASONING_EFFORT
    }
    return defaultReasoningEffortForModel(
      backendConfig,
      promptEnhancerModel,
      promptEnhancerReasoningEffortOptions,
    )
  }, [
    backendConfig,
    promptEnhancerModel,
    promptEnhancerReasoningEffort,
    promptEnhancerReasoningEffortOptions,
  ])

  const persistPromptEnhancerReasoningEffort = useCallback(
    (effort: PromptEnhancerReasoningEffort) => {
      setPromptEnhancerReasoningEffort(effort)
      try {
        window.localStorage.setItem(
          PROMPT_ENHANCER_REASONING_EFFORT_STORAGE_KEY,
          effort,
        )
      } catch {
        // Storage can be unavailable in restricted browsers.
      }
    },
    [],
  )

  const persistPromptEnhancerSeed = useCallback(
    (seed: PromptEnhancerSeed | undefined) => {
      setPromptEnhancerSeed(seed)
      try {
        if (seed === undefined) {
          window.localStorage.removeItem(PROMPT_ENHANCER_SEED_STORAGE_KEY)
        } else {
          window.localStorage.setItem(
            PROMPT_ENHANCER_SEED_STORAGE_KEY,
            String(seed),
          )
        }
      } catch {
        // Storage can be unavailable in restricted browsers.
      }
    },
    [],
  )

  const selectInferenceSteps = useCallback((steps: InferenceSteps) => {
    setInferenceSteps(steps)
    try {
      window.localStorage.setItem(
        INFERENCE_STEPS_STORAGE_KEY,
        String(steps),
      )
    } catch {
      // Storage can be unavailable in restricted browsers.
    }
  }, [])

  useEffect(() => {
    if (!backendConfig) return
    if (inferenceSteps !== resolvedInferenceSteps) {
      selectInferenceSteps(resolvedInferenceSteps)
    }
  }, [
    backendConfig,
    inferenceSteps,
    resolvedInferenceSteps,
    selectInferenceSteps,
  ])

  const updateWorldConfig = useCallback((nextConfig: WorldConfig) => {
    const perspective = normalizePerspective(nextConfig.perspective)
    setConfig({ ...nextConfig, perspective })
    try {
      window.localStorage.setItem(PERSPECTIVE_STORAGE_KEY, perspective)
    } catch {
      // Storage can be unavailable in restricted browsers.
    }
  }, [])

  useEffect(() => {
    if (!backendConfig) return
    if (promptEnhancerReasoningEffort !== resolvedPromptEnhancerReasoningEffort) {
      persistPromptEnhancerReasoningEffort(
        resolvedPromptEnhancerReasoningEffort,
      )
    }
  }, [
    backendConfig,
    persistPromptEnhancerReasoningEffort,
    promptEnhancerReasoningEffort,
    resolvedPromptEnhancerReasoningEffort,
  ])

  const openStudio = useCallback(() => {
    setSessionNotice(undefined)
    setDryRun(false)
    setScreen('studio')
  }, [])

  const returnToSetup = useCallback(() => {
    logoutAfterWorldReleaseRef.current = false
    setSessionNotice(undefined)
    setDryRun(false)
    setConfig(initialWorldConfig())
    setScreen('setup')
  }, [])

  const expireWorldSession = useCallback((message?: string) => {
    logoutAfterWorldReleaseRef.current = false
    setSessionNotice(
      message
      ?? 'Your world session expired after 15 minutes of inactivity. Create a new world to continue.',
    )
    setDryRun(false)
    setConfig(initialWorldConfig())
    setScreen('setup')
  }, [])

  const releaseHostLease = useCallback(async () => {
    if (!multiHost) return
    await fetch(`${import.meta.env.BASE_URL}auth/logout`, {
      method: 'POST',
      credentials: 'same-origin',
      keepalive: true,
    }).catch(() => undefined)
    setDispatchSession(undefined)
    setScreen('setup')
    setBackendConfig(undefined)
  }, [multiHost])

  const finishWorldSessionRelease = useCallback(() => {
    if (!logoutAfterWorldReleaseRef.current) return
    logoutAfterWorldReleaseRef.current = false
    void releaseHostLease()
  }, [releaseHostLease])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    document.documentElement.style.colorScheme = theme
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute('content', theme === 'dark' ? '#11130f' : '#f6f0e6')
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme)
    } catch {
      // Theme persistence is optional in privacy-restricted browsers.
    }
  }, [theme])

  useEffect(() => {
    if (!multiHost) return
    let cancelled = false
    const check = async () => {
      if (pageClosingRef.current) return
      try {
        const response = await fetch(`${import.meta.env.BASE_URL}auth/status`, {
          cache: 'no-store',
          credentials: 'same-origin',
          headers: { 'X-Nomas-Page-Id': dispatcherPageIdRef.current },
        })
        if (cancelled) return
        if (response.status === 401) {
          setDispatchSession(undefined)
          return
        }
        const value = await parseJsonResponse<DispatchSession>(
          response,
          'GET auth/status',
        )
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        setDispatchSession(value)
        setLoginError(undefined)
      } catch (error) {
        if (!cancelled) {
          setLoginError(error instanceof Error ? error.message : String(error))
        }
      }
    }
    const reconnect = async () => {
      if (pageClosingRef.current) return
      try {
        const response = await fetch(
          `${import.meta.env.BASE_URL}auth/reconnect`,
          {
            method: 'POST',
            cache: 'no-store',
            credentials: 'same-origin',
            headers: { 'X-Nomas-Page-Id': dispatcherPageIdRef.current },
          },
        )
        if (cancelled) return
        if (response.status === 401) {
          setDispatchSession(undefined)
          return
        }
        const value = await parseJsonResponse<DispatchSession>(
          response,
          'POST auth/reconnect',
        )
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        setDispatchSession(value)
        setLoginError(undefined)
      } catch (error) {
        if (!cancelled) {
          setLoginError(error instanceof Error ? error.message : String(error))
        }
      }
    }
    void reconnect()
    const timer = window.setInterval(check, 5000)
    const reconnectOnPageShow = () => {
      if (pageClosingRef.current) {
        // A BFCache restore keeps the same JavaScript context, so rotate the
        // page id before reconnecting. Delayed requests carrying the closed
        // page id can no longer revive its lease.
        dispatcherPageIdRef.current = createDispatcherPageId()
      }
      pageClosingRef.current = false
      void reconnect()
    }
    window.addEventListener('pageshow', reconnectOnPageShow)
    return () => {
      cancelled = true
      window.clearInterval(timer)
      window.removeEventListener('pageshow', reconnectOnPageShow)
    }
  }, [multiHost])

  useEffect(() => {
    if (!dispatchSession) return
    let lastHeartbeatAt = 0
    const heartbeat = () => {
      if (pageClosingRef.current) return
      const now = Date.now()
      if (now - lastHeartbeatAt < 30_000) return
      lastHeartbeatAt = now
      void fetch(`${import.meta.env.BASE_URL}auth/heartbeat`, {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'X-Nomas-Page-Id': dispatcherPageIdRef.current },
      })
    }
    heartbeat()
    window.addEventListener('pointerdown', heartbeat)
    window.addEventListener('keydown', heartbeat)
    window.addEventListener('touchstart', heartbeat, { passive: true })
    window.addEventListener('wheel', heartbeat, { passive: true })
    return () => {
      window.removeEventListener('pointerdown', heartbeat)
      window.removeEventListener('keydown', heartbeat)
      window.removeEventListener('touchstart', heartbeat)
      window.removeEventListener('wheel', heartbeat)
    }
  }, [dispatchSession?.status])

  useEffect(() => {
    if (!multiHost) return
    let cancelled = false
    const loadDispatcherHealth = () => {
      void fetch(`${import.meta.env.BASE_URL}health`, {
        cache: 'no-store',
      })
        .then((response) => parseJsonResponse<DispatcherHealth>(
          response,
          'GET health',
        ))
        .then((value) => {
          if (!cancelled) setDispatcherHealth(value)
        })
        .catch(() => undefined)
    }
    loadDispatcherHealth()
    const timer = window.setInterval(loadDispatcherHealth, 5000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [multiHost])

  const visibleBackendConfig = backendConfig
    ? {
        ...backendConfig,
        ...(multiHost && dispatcherHealth?.hosts
          ? {
              gpu_node_total: dispatcherHealth.hosts.length,
              gpu_node_online: dispatcherHealth.hosts.filter(
                (host) => host.online,
              ).length,
              gpu_node_available: dispatcherHealth.hosts.filter(
                (host) => host.online && host.available,
              ).length,
              dispatcher_active_users: dispatcherHealth.active_users ?? 0,
              dispatcher_waitlist_size: dispatcherHealth.waitlist_size ?? 0,
            }
          : {}),
      }
    : backendConfig

  useEffect(() => {
    if (!multiHost || !dispatchSession) return
    const disconnect = () => {
      pageClosingRef.current = true
      const url = `${import.meta.env.BASE_URL}auth/disconnect`
      const body = JSON.stringify({ page_id: dispatcherPageIdRef.current })
      if (typeof navigator.sendBeacon === 'function') {
        navigator.sendBeacon(
          url,
          new Blob([body], { type: 'application/json' }),
        )
        return
      }
      void fetch(url, {
        method: 'POST',
        credentials: 'same-origin',
        keepalive: true,
        headers: { 'Content-Type': 'application/json' },
        body,
      }).catch(() => undefined)
    }
    window.addEventListener('pagehide', disconnect)
    return () => window.removeEventListener('pagehide', disconnect)
  }, [dispatchSession?.status, multiHost])

  const login = async () => {
    if (!loginName.trim()) return
    setLoggingIn(true)
    setLoginError(undefined)
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}auth/login`, {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: loginName.trim(),
          page_id: dispatcherPageIdRef.current,
        }),
      })
      const value = await parseJsonResponse<
        DispatchSession | { detail?: string }
      >(response, 'POST auth/login')
      if (!response.ok) {
        throw new Error('detail' in value ? String(value.detail) : `HTTP ${response.status}`)
      }
      setDispatchSession(value as DispatchSession)
    } catch (error) {
      setLoginError(error instanceof Error ? error.message : String(error))
    } finally {
      setLoggingIn(false)
    }
  }

  const logout = async () => {
    if (screen === 'studio') {
      // Unmounting WorldStudio first lets it release the backend's
      // process-local interactive session before the dispatcher cookie and
      // host lease are removed.
      logoutAfterWorldReleaseRef.current = true
      setScreen('setup')
      return
    }
    await releaseHostLease()
  }

  const loadAllocations = async () => {
    allocationsRequestRef.current?.abort()
    const controller = new AbortController()
    allocationsRequestRef.current = controller
    const isCurrentRequest = () =>
      mountedRef.current
      && allocationsRequestRef.current === controller
      && !controller.signal.aborted

    setAllocationsLoading(true)
    setAllocationsError(undefined)
    try {
      for (
        let attempt = 0;
        attempt < ALLOCATIONS_FETCH_MAX_ATTEMPTS;
        attempt += 1
      ) {
        try {
          const response = await fetch(
            `${import.meta.env.BASE_URL}auth/allocations`,
            {
              cache: 'no-store',
              credentials: 'same-origin',
              signal: controller.signal,
            },
          )
          if (
            ALLOCATIONS_RETRYABLE_STATUSES.has(response.status)
            && attempt + 1 < ALLOCATIONS_FETCH_MAX_ATTEMPTS
          ) {
            await waitForRetry(
              ALLOCATIONS_FETCH_RETRY_BASE_MS * (2 ** attempt),
              controller.signal,
            )
            continue
          }
          const value = await parseJsonResponse<
            ActiveAllocationsResponse | { detail?: string }
          >(response, 'GET auth/allocations')
          if (!response.ok) {
            throw new Error(
              'detail' in value ? String(value.detail) : `HTTP ${response.status}`,
            )
          }
          if (isCurrentRequest()) {
            setAllocations(value as ActiveAllocationsResponse)
            setAllocationsError(undefined)
          }
          return
        } catch (error) {
          if (isAbortError(error)) return
          if (
            error instanceof TypeError
            && attempt + 1 < ALLOCATIONS_FETCH_MAX_ATTEMPTS
          ) {
            await waitForRetry(
              ALLOCATIONS_FETCH_RETRY_BASE_MS * (2 ** attempt),
              controller.signal,
            )
            continue
          }
          throw error
        }
      }
    } catch (error) {
      if (isCurrentRequest() && !isAbortError(error)) {
        setAllocationsError(
          error instanceof Error ? error.message : String(error),
        )
      }
    } finally {
      if (isCurrentRequest()) {
        setAllocationsLoading(false)
        allocationsRequestRef.current = undefined
      }
    }
  }

  const openAllocations = () => {
    setShowAllocations(true)
    void loadAllocations()
  }

  const addGalleryItem = useCallback((item: GalleryItem) => {
    setBackendConfig((current) => {
      if (!current) return current
      const previous = current.gallery_items ?? []
      return {
        ...current,
        gallery_items: [
          item,
          ...previous.filter((candidate) => candidate.id !== item.id),
        ],
        gallery_hidden_ids: (current.gallery_hidden_ids ?? []).filter(
          (candidate) => candidate !== item.id,
        ),
      }
    })
  }, [])

  const removeGalleryItem = useCallback((itemId: string) => {
    setBackendConfig((current) => {
      if (!current) return current
      return {
        ...current,
        gallery_items: (current.gallery_items ?? []).filter(
          (candidate) => candidate.id !== itemId,
        ),
        gallery_hidden_ids: [
          ...(current.gallery_hidden_ids ?? []).filter(
            (candidate) => candidate !== itemId,
          ),
          itemId,
        ],
      }
    })
  }, [])

  useEffect(() => {
    const handleGalleryItemRemoved = (event: Event) => {
      const itemId = (
        event as CustomEvent<GalleryItemRemovedDetail>
      ).detail?.itemId
      if (itemId) removeGalleryItem(itemId)
    }
    window.addEventListener(
      GALLERY_ITEM_REMOVED_EVENT,
      handleGalleryItemRemoved,
    )
    return () => {
      window.removeEventListener(
        GALLERY_ITEM_REMOVED_EVENT,
        handleGalleryItemRemoved,
      )
    }
  }, [removeGalleryItem])

  useEffect(() => {
    if (STUDIO_PREVIEW) return
    if (multiHost && dispatchSession?.status !== 'assigned') {
      return
    }
    const controller = new AbortController()
    let left = false
    const leave = () => {
      if (left) return
      left = true
      void fetch(
        `${import.meta.env.BASE_URL}v1/presence/${encodeURIComponent(browserClientId)}`,
        {
          method: 'DELETE',
          keepalive: true,
        },
      ).catch(() => undefined)
    }

    const loadConfig = () => {
      void fetch(`${import.meta.env.BASE_URL}v1/config`, {
        headers: {
          Accept: 'application/json',
          'X-Nomas-Client-Id': browserClientId,
        },
        cache: 'no-store',
        signal: controller.signal,
      })
        .then(async (response) => {
          const contentType = response.headers.get('content-type') ?? ''
          const rawBody = await response.text()
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${rawBody.slice(0, 200)}`)
          }
          if (!contentType.toLowerCase().includes('application/json')) {
            throw new Error(`Expected JSON but received ${contentType || 'unknown content type'}`)
          }
          return JSON.parse(rawBody) as BackendConfig
        })
        .then((value) => {
          setBackendConfig((current) => {
            if (!current?.gallery_items?.length || !value.gallery_items?.length) {
              return value
            }
            const currentItems = new Map(
              current.gallery_items.map((item) => [item.id, item]),
            )
            return {
              ...value,
              gallery_items: value.gallery_items.map((item) => {
                const previous = currentItems.get(item.id)
                return previous?.prompt_timeline?.length
                  && !item.prompt_timeline?.length
                  ? {
                      ...item,
                      prompt_timeline: previous.prompt_timeline,
                    }
                  : item
              }),
            }
          })
          setBackendError(undefined)
          console.log('[Hy-Nomas] Backend config:', value)
          if (multiHost) loadGallery()
        })
        .catch((error: unknown) => {
          if (error instanceof DOMException && error.name === 'AbortError') return
          const message = error instanceof Error ? error.message : String(error)
          setBackendError(message)
          console.error('[Hy-Nomas] Failed to load backend config:', error)
        })
    }

    const loadGallery = () => {
      void fetch(`${import.meta.env.BASE_URL}v1/gallery`, {
        headers: { Accept: 'application/json' },
        cache: 'no-store',
        signal: controller.signal,
      })
        .then(async (response) => {
          if (!response.ok) throw new Error(`HTTP ${response.status}`)
          return response.json() as Promise<{
            items?: GalleryItem[]
            hidden_ids?: string[]
          }>
        })
        .then((value) => {
          if (!Array.isArray(value.items)) return
          setBackendConfig((current) =>
            current
              ? {
                  ...current,
                  gallery_items: value.items,
                  gallery_hidden_ids: Array.isArray(value.hidden_ids)
                    ? value.hidden_ids
                    : current.gallery_hidden_ids,
                }
              : current,
          )
        })
        .catch((error: unknown) => {
          if (error instanceof DOMException && error.name === 'AbortError') return
          console.warn('[Hy-Nomas] Failed to refresh Gallery:', error)
        })
    }

    loadConfig()
    const timer = window.setInterval(loadConfig, 5000)
    window.addEventListener('pagehide', leave)

    return () => {
      window.clearInterval(timer)
      window.removeEventListener('pagehide', leave)
      controller.abort()
      leave()
    }
  }, [dispatchSession?.status, multiHost])

  if (multiHost && !dispatchSession) {
    return (
      <main className="access-shell">
        <form className="access-card" onSubmit={(event) => { event.preventDefault(); void login() }}>
          <img src={`${import.meta.env.BASE_URL}hunyuan-logo.png`} alt="" />
          <span>HY-NOMAS</span>
          <h1>Sign in</h1>
          <p>One user receives exclusive access to one 16-GPU world slot.</p>
          <input
            aria-label="User ID"
            value={loginName}
            onChange={(event) => setLoginName(event.target.value)}
            placeholder="User ID"
            autoFocus
          />
          {loginError && <small className="access-error">{loginError}</small>}
          <button className="primary-button" disabled={loggingIn || !loginName.trim()}>
            {loggingIn ? 'Signing in…' : 'Continue'}
          </button>
        </form>
      </main>
    )
  }

  if (dispatchSession?.status === 'waiting') {
    return (
      <main className="access-shell">
        <section className="access-card waitlist-card">
          <span>GPU HOST WAITLIST</span>
          <h1>#{dispatchSession.waitlist_position ?? '—'}</h1>
          <p>
            You are in line. {dispatchSession.waitlist_size} user
            {dispatchSession.waitlist_size === 1 ? '' : 's'} currently waiting.
            This page updates automatically.
          </p>
          <button type="button" className="secondary-button" onClick={() => void logout()}>
            Leave waitlist
          </button>
        </section>
      </main>
    )
  }

  return <div className="leased-app">
    {dispatchSession && (
      <div className="lease-banner">
        <span>Dedicated SP16 slot: {dispatchSession.host_id}</span>
        <div className="lease-actions">
          <button
            type="button"
            className="allocations-button"
            onClick={openAllocations}
          >
            Active users
          </button>
          <button
            type="button"
            className="logout-button"
            onClick={() => void logout()}
          >
            Log out
          </button>
        </div>
      </div>
    )}
    {showAllocations && (
      <div
        className="allocations-backdrop"
        role="presentation"
        onMouseDown={(event) => {
          if (event.target === event.currentTarget) setShowAllocations(false)
        }}
      >
        <section
          className="allocations-dialog"
          role="dialog"
          aria-modal="true"
          aria-labelledby="allocations-title"
        >
          <header>
            <div>
              <span>RESOURCE ALLOCATIONS</span>
              <h2 id="allocations-title">Active users</h2>
            </div>
            <button
              type="button"
              className="allocations-close"
              aria-label="Close active users"
              onClick={() => setShowAllocations(false)}
            >
              ×
            </button>
          </header>
          <div className="allocations-toolbar">
            <span>
              {allocations
                ? `${allocations.count} active allocation${allocations.count === 1 ? '' : 's'}`
                : 'Loading allocations…'}
            </span>
            <button
              type="button"
              disabled={allocationsLoading}
              onClick={() => void loadAllocations()}
            >
              {allocationsLoading ? 'Refreshing…' : 'Refresh'}
            </button>
          </div>
          {allocationsError && (
            <p className="allocations-error">{allocationsError}</p>
          )}
          {allocations?.allocations.length ? (
            <div className="allocations-table-wrap">
              <table className="allocations-table">
                <thead>
                  <tr>
                    <th>User ID</th>
                    <th>GPU host</th>
                  </tr>
                </thead>
                <tbody>
                  {allocations.allocations.map((allocation) => (
                    <tr key={`${allocation.host_id}:${allocation.user_id}`}>
                      <td>{allocation.user_id}</td>
                      <td>{allocation.host_id}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : allocations ? (
            <p className="allocations-empty">No GPU hosts are currently assigned.</p>
          ) : allocationsLoading ? (
            <p className="allocations-empty">Loading…</p>
          ) : null}
        </section>
      </div>
    )}
    {screen === 'setup' ? (
    <SetupScreen
      config={config}
      backendConfig={visibleBackendConfig}
      backendError={backendError}
      sessionNotice={sessionNotice}
      promptEnhancerModel={promptEnhancerModel}
      onPromptEnhancerModelChange={selectPromptEnhancerModel}
      promptEnhancerReasoningEffort={resolvedPromptEnhancerReasoningEffort}
      promptEnhancerReasoningEffortOptions={
        promptEnhancerReasoningEffortOptions
      }
      onPromptEnhancerReasoningEffortChange={
        persistPromptEnhancerReasoningEffort
      }
      promptEnhancerSeed={promptEnhancerSeed}
      promptEnhancerSeedMin={promptEnhancerSeedRangeValue.min}
      promptEnhancerSeedMax={promptEnhancerSeedRangeValue.max}
      onPromptEnhancerSeedChange={persistPromptEnhancerSeed}
      inferenceSteps={resolvedInferenceSteps}
      inferenceStepsOptions={availableInferenceSteps}
      onInferenceStepsChange={selectInferenceSteps}
      onChange={updateWorldConfig}
      onStart={openStudio}
      theme={theme}
      onThemeToggle={() =>
        setTheme((current) => current === 'dark' ? 'light' : 'dark')
      }
    />
  ) : (
    <WorldStudio
      config={config}
      backendConfig={visibleBackendConfig}
      dryRun={dryRun}
      promptEnhancerModel={promptEnhancerModel}
      onPromptEnhancerModelChange={selectPromptEnhancerModel}
      promptEnhancerReasoningEffort={resolvedPromptEnhancerReasoningEffort}
      promptEnhancerReasoningEffortOptions={
        promptEnhancerReasoningEffortOptions
      }
      onPromptEnhancerReasoningEffortChange={
        persistPromptEnhancerReasoningEffort
      }
      promptEnhancerSeed={promptEnhancerSeed}
      promptEnhancerSeedMin={promptEnhancerSeedRangeValue.min}
      promptEnhancerSeedMax={promptEnhancerSeedRangeValue.max}
      onPromptEnhancerSeedChange={persistPromptEnhancerSeed}
      inferenceSteps={resolvedInferenceSteps}
      inferenceStepsOptions={availableInferenceSteps}
      onInferenceStepsChange={selectInferenceSteps}
      onReconfigure={returnToSetup}
      onSessionExpired={expireWorldSession}
      onSessionReleased={finishWorldSessionRelease}
      onGalleryItemAdded={addGalleryItem}
      theme={theme}
      onThemeToggle={() =>
        setTheme((current) => current === 'dark' ? 'light' : 'dark')
      }
    />
    )}
  </div>
}

export default App

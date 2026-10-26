export type Perspective =
  | 'first-person'
  | 'third-person follow'
  | 'second-person'
  | 'third-person observer'

export type LegacyPerspective = Perspective | 'third-person'

export type InferenceSteps = number

export type WorldStyle =
  | 'real-world footage'
  | 'animation/anime'
  | 'CG'
  | 'video game'
  | 'software/web/app/computer screen recording'
  | 'slideshow'

export interface WorldConfig {
  perspective: Perspective
  subjectDescription: string
  environmentDescription: string
}

export type PromptEnhancerModel = string

export type PromptEnhancerReasoningEffort =
  | 'none'
  | 'low'
  | 'medium'
  | 'high'
  | 'xhigh'

export type PromptEnhancerSeed = number

export interface PromptEnhancerReasoningEffortOption {
  id: PromptEnhancerReasoningEffort
  label: string
  default?: boolean
}

export interface PromptEnhancerOption {
  id: PromptEnhancerModel
  label: string
  default?: boolean
  reasoning_effort_options?: Array<
    PromptEnhancerReasoningEffort | PromptEnhancerReasoningEffortOption
  >
  default_reasoning_effort?: PromptEnhancerReasoningEffort
}

export interface GalleryItem {
  id: string
  title: string
  video: string
  duration: string
  perspective: LegacyPerspective
  subject: string
  attributes: string
  direction: string
  prompt_timeline?: Array<{
    number: number
    prompt: string
    start_seconds: number
    end_seconds: number
  }>
  created_at?: number | null
}

export const GALLERY_ITEM_REMOVED_EVENT = 'nomas:gallery-item-removed'

export interface GalleryItemRemovedDetail {
  itemId: string
}

export interface BackendConfig {
  rollout_type: 'seg_ar' | 'oneshot'
  prompt_source: 'prompt' | 'subject_file'
  num_frames: number | null
  num_frames_source: 'request_or_config' | 'prompt_timestamps'
  fps: number
  trajectory_translation_scale?: number
  trajectory_rotation_scale?: number
  num_inference_steps?: number
  num_inference_steps_options?: number[]
  default_num_inference_steps?: number
  model_name?: string
  dit_weight?: string
  dry_run_enabled?: boolean
  dry_run_source?: string | null
  architecture?: 'gateway_worker'
  worker_online?: boolean
  worker_status?: 'offline' | 'ready' | 'busy' | 'reloading' | string
  worker_count?: number
  online_worker_count?: number
  busy_worker_count?: number
  queued_jobs?: number
  running_jobs?: number
  reserved_session_count?: number
  concurrency_enabled?: boolean
  online_users?: number
  resources?: ResourceSnapshot
  recent_errors?: DemoErrorEvent[]
  prompt_enhancement_enabled?: boolean
  prompt_enhancement_profile?: 'markdown' | null
  prompt_enhancement_profile_sha256?: string | null
  prompt_enhancement_system_sha256?: string | null
  prompt_enhancer_options?: PromptEnhancerOption[]
  default_prompt_enhancer_model?: PromptEnhancerModel
  prompt_enhancer_reasoning_effort_options?:
    | Array<
      PromptEnhancerReasoningEffort | PromptEnhancerReasoningEffortOption
    >
    | Partial<Record<
      PromptEnhancerModel,
      Array<
        PromptEnhancerReasoningEffort | PromptEnhancerReasoningEffortOption
      >
    >>
  default_prompt_enhancer_reasoning_effort?:
    | PromptEnhancerReasoningEffort
    | Partial<Record<
      PromptEnhancerModel,
      PromptEnhancerReasoningEffort
    >>
  prompt_enhancer_seed_min?: number
  prompt_enhancer_seed_max?: number
  prompt_fallback_enabled?: boolean
  gallery_items?: GalleryItem[]
  gallery_hidden_ids?: string[]
  gpu_node_total?: number
  gpu_node_online?: number
  gpu_node_available?: number
  dispatcher_active_users?: number
  dispatcher_waitlist_size?: number
}

export interface GpuResource {
  index: number
  name: string
  gpu_utilization_percent: number
  memory_utilization_percent: number
  memory_used_mib: number
  memory_total_mib: number
  power_draw_watts: number
  power_limit_watts: number
  power_ratio: number
  sm_clock_mhz: number
  pstate: string
}

export interface ResourceSnapshot {
  available: boolean
  sampled_at?: number | null
  online_users: number
  presence_ttl_seconds?: number
  total_power_draw_watts?: number
  total_power_limit_watts?: number
  average_power_ratio?: number
  average_gpu_utilization_percent?: number
  memory_used_mib?: number
  memory_total_mib?: number
  max_gpu_memory_ratio?: number
  memory_pressure?: 'normal' | 'warning' | 'critical'
  compute_active?: boolean
  compute_state?: 'active' | 'scheduled' | 'idle'
  gpus: GpuResource[]
  error?: string
}

export interface DemoErrorEvent {
  id: string
  code: string
  message: string
  job_id?: string | null
  worker_id?: string | null
  recoverable: boolean
  created_at: number
}

export interface DispatchSession {
  user_id: string
  status: 'assigned' | 'waiting'
  host_id?: string | null
  waitlist_position?: number | null
  waitlist_size: number
  lease_seconds: number
  lease_expires_at: number
  server_time: number
}

export interface ActiveAllocation {
  user_id: string
  host_id: string
  assigned_at?: number | null
  last_seen_at?: number | null
}

export interface ActiveAllocationsResponse {
  allocations: ActiveAllocation[]
  count: number
  server_time: number
}

export interface DryRunJob {
  id: string
  status: 'queued' | 'running' | 'succeeded' | 'failed'
  sample_id: string
  source: string
  captions: string[]
  total_segments: number
  completed_segments: number
  queue_position?: number
  queue_reason?: string
  video_url?: string | null
  error?: string | null
  segment_updates: Array<{
    segment_index: number
    caption: string
    video_url: string
    start_frame?: number
    end_frame?: number
    playback_start_seconds?: number
    playback_end_seconds?: number
  }>
  result?: {
    id: string
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
  }
}

export interface GenerateJob {
  id: string
  status: 'enhancing' | 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'
  queue_position?: number
  queue_reason?: string
  pe_prompt?: string
  pe_reasoning?: string
  pe_content_preview?: string
  pe_content_started?: boolean
  pe_request_id?: string | null
  prompt_enhancement_mode?: string
  prompt_enhancer_reasoning_effort?: PromptEnhancerReasoningEffort
  prompt_enhancer_seed?: PromptEnhancerSeed
  error?: string | null
  error_code?: string | null
  result?: {
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
    latent_sha256?: string
  }
}

export interface RollbackJob {
  id: string
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'
  queue_position?: number
  error?: string | null
  result?: {
    session_id: string
    remaining_segments: number
    removed_cache_tokens: number
    removed_latent_frames: number
    video_url?: string | null
    frames: number
    fps: number
    stream_init_url?: string | null
    stream_segment_url?: string | null
    stream_sequence?: number | null
    stream_mime_type?: string | null
  }
}

export interface ChatMessage {
  id: number
  role: 'agent' | 'user'
  content: string
  meta?: string
  cameraLabel?: string
  thinking?: string
}

export interface CameraFrame {
  frame: number
  t: number
  position: { x: number; y: number; z: number }
  quaternion: { x: number; y: number; z: number; w: number }
  fov: number
}

export type CameraControlKey =
  | 'KeyW'
  | 'KeyA'
  | 'KeyS'
  | 'KeyD'
  | 'ArrowUp'
  | 'ArrowDown'
  | 'ArrowLeft'
  | 'ArrowRight'

export interface CameraKeyEvent {
  t: number
  code: CameraControlKey
  pressed: boolean
}

export interface CameraPath {
  /** Generation-time user setting, used only to derive the 0.75x visual gain. */
  visualizationRotationScale?: number
  id: string
  name: string
  source: 'preset' | 'recorded' | 'imported'
  trajectory: string
  durationMs?: number
  sampleRate?: number
  totalDistance?: number
  frames?: CameraFrame[]
  keyEvents?: CameraKeyEvent[]
}

export interface SelectOption<T extends string> {
  value: T
  label: string
  description: string
}

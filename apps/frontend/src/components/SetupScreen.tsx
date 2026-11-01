import {
  ArrowRight,
  Bot,
  Images,
  MoreHorizontal,
  Orbit,
} from 'lucide-react'
import { useEffect, useRef, useState, type FormEvent } from 'react'
import type {
  BackendConfig,
  InferenceSteps,
  Perspective,
  PromptEnhancerModel,
  PromptEnhancerReasoningEffort,
  PromptEnhancerReasoningEffortOption,
  PromptEnhancerSeed,
  SelectOption,
  WorldConfig,
} from '../types'
import type { GallerySample } from '../gallerySamples'
import { normalizePerspective } from '../perspective'
import { WorldGallery } from './WorldGallery'
import { ResourceStatus } from './ResourceStatus'
import { ThemeToggle } from './ThemeToggle'

const perspectives: Array<SelectOption<Perspective>> = [
  {
    value: 'third-person follow',
    label: 'Third person',
    description: 'Camera follows behind or alongside the main subject.',
  },
  {
    value: 'first-person',
    label: 'First person',
    description: 'Camera sees directly through the subject’s eyes.',
  },
  {
    value: 'second-person',
    label: 'Second person (talking head)',
    description: 'Subject faces and looks directly into the camera.',
  },
  {
    value: 'third-person observer',
    label: 'Third person observer',
    description: 'External observer view that does not follow the subject.',
  },
]

const subjectTemplates = [
  {
    label: 'Astronaut',
    value: 'A lone astronaut in a weathered white pressure suit',
  },
  {
    label: 'Clockwork fox',
    value: 'A small brass clockwork fox with green glass eyes and a red scarf',
  },
  {
    label: 'Station wagon',
    value: 'A blue station wagon with chrome trim and worn paint',
  },
]

const environmentTemplates = [
  {
    label: 'Floating greenhouse',
    value: 'A glass greenhouse floats above a stormy ocean at blue hour.',
  },
  {
    label: 'Mossy observatory',
    value: 'A moss-covered observatory at dawn, with tall windows and a rotating brass orrery.',
  },
  {
    label: 'Desert road',
    value: 'A winding desert road crosses red sandstone mountains beneath a clear sky.',
  },
]

interface SetupScreenProps {
  config: WorldConfig
  backendConfig?: BackendConfig
  backendError?: string
  sessionNotice?: string
  promptEnhancerModel: PromptEnhancerModel
  onPromptEnhancerModelChange: (model: PromptEnhancerModel) => void
  promptEnhancerReasoningEffort: PromptEnhancerReasoningEffort
  promptEnhancerReasoningEffortOptions: PromptEnhancerReasoningEffortOption[]
  onPromptEnhancerReasoningEffortChange: (
    effort: PromptEnhancerReasoningEffort,
  ) => void
  promptEnhancerSeed?: PromptEnhancerSeed
  promptEnhancerSeedMin: number
  promptEnhancerSeedMax: number
  onPromptEnhancerSeedChange: (seed: PromptEnhancerSeed | undefined) => void
  inferenceSteps: InferenceSteps
  inferenceStepsOptions: InferenceSteps[]
  onInferenceStepsChange: (steps: InferenceSteps) => void
  onChange: (config: WorldConfig) => void
  onStart: () => void
  theme: 'dark' | 'light'
  onThemeToggle: () => void
}

export function SetupScreen({
  config,
  backendConfig,
  backendError,
  sessionNotice,
  promptEnhancerModel,
  onPromptEnhancerModelChange,
  promptEnhancerReasoningEffort,
  promptEnhancerReasoningEffortOptions,
  onPromptEnhancerReasoningEffortChange,
  promptEnhancerSeed,
  promptEnhancerSeedMin,
  promptEnhancerSeedMax,
  onPromptEnhancerSeedChange,
  inferenceSteps,
  inferenceStepsOptions,
  onInferenceStepsChange,
  onChange,
  onStart,
  theme,
  onThemeToggle,
}: SetupScreenProps) {
  const [moreMenuOpen, setMoreMenuOpen] = useState(false)
  const [imageUploading, setImageUploading] = useState(false)
  const [imageError, setImageError] = useState<string>()
  const uploadInputRef = useRef<HTMLInputElement>(null)
  const imageRequestRef = useRef(0)
  const manualConfigRef = useRef<WorldConfig | undefined>(undefined)
  const configRef = useRef(config)
  configRef.current = config
  const moreMenuRef = useRef<HTMLDivElement>(null)
  const isOneShot = backendConfig?.rollout_type === 'oneshot'
  const imageControlsLocked = !isOneShot && (config.referenceMode === 'image' || Boolean(config.referenceImage))
  const captionReady = !imageControlsLocked || config.referenceImage?.caption_status === 'ready'
  const setupInputReady = imageControlsLocked
    ? Boolean(config.imageContinuationPrompt?.trim())
    : Boolean(config.subjectDescription.trim() && (isOneShot || config.environmentDescription.trim()))
  const update = <Key extends keyof WorldConfig>(
    key: Key,
    value: WorldConfig[Key],
  ) => {
    if (imageControlsLocked && (key === 'subjectDescription' || key === 'environmentDescription')) return
    onChange({ ...config, [key]: value })
  }

  const selectGallerySample = (sample: GallerySample) => {
    if (imageControlsLocked) return
    onChange({
      ...config,
      subjectDescription: sample.subject,
      environmentDescription: [sample.attributes, sample.direction]
        .filter((value) => value.trim())
        .join(' '),
      perspective: normalizePerspective(sample.perspective),
    })
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (
      !imageUploading
      && captionReady
      && setupInputReady
    ) {
      onStart()
    }
  }

  const uploadImage = async (file: File | undefined, standardReference = false) => {
    if (!file) return
    // Browser MIME labels can be empty or use image/jpg for valid JPEGs.
    // The server validates the actual image bytes before accepting the upload.
    const supportedImage = ['image/jpeg', 'image/jpg', 'image/pjpeg', 'image/png', 'image/webp'].includes(file.type)
      || /\.(jpe?g|jfif|png|webp)$/i.test(file.name)
    if (!supportedImage
      || file.size > 10 * 1024 * 1024) {
      setImageError('Choose a JPEG, PNG, or WebP image up to 10 MB.')
      return
    }
    const request = ++imageRequestRef.current
    if (standardReference && !imageControlsLocked) manualConfigRef.current = { ...configRef.current }
    if (standardReference) onChange({ ...configRef.current, referenceMode: 'image', referenceImage: undefined,
      subjectDescription: '', environmentDescription: '' })
    setImageUploading(true)
    setImageError(undefined)
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}v1/reference-images?perspective=${encodeURIComponent(configRef.current.perspective)}`, {
        method: 'POST', headers: { 'Content-Type': file.type || 'application/octet-stream' }, body: file,
      })
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail || 'Image upload failed')
      if (request !== imageRequestRef.current) return
      const pending: WorldConfig = { ...configRef.current, referenceMode: 'image', referenceImage: result, subjectDescription: '', environmentDescription: '' }
      onChange(pending)
      const deadline = Date.now() + 6 * 60 * 1000
      while (request === imageRequestRef.current) {
        const captionResponse = await fetch(`${import.meta.env.BASE_URL}v1/reference-images/${result.id}`)
        const caption = await captionResponse.json()
        if (request !== imageRequestRef.current) return
        if (!captionResponse.ok || caption.caption_status === 'failed') {
          throw new Error(caption.caption_error || caption.detail || 'Image captioning failed')
        }
        if (caption.caption_status === 'ready') {
          onChange({
            ...configRef.current,
            referenceMode: 'image',
            referenceImage: { ...result, caption_status: 'ready' },
            subjectDescription: caption.subject_description,
            environmentDescription: caption.environment_description,
          })
          break
        }
        if (Date.now() >= deadline) throw new Error('Image captioning timed out. Remove the image and try again.')
        await new Promise((resolve) => window.setTimeout(resolve, 1500))
      }
    } catch (error) {
      if (request === imageRequestRef.current) {
        setImageError(error instanceof Error ? error.message : 'Image upload failed')
      }
    } finally {
      if (request === imageRequestRef.current) setImageUploading(false)
    }
  }

  const removeImage = () => {
    ++imageRequestRef.current
    setImageUploading(false)
    setImageError(undefined)
    const current = configRef.current
    onChange({ ...(manualConfigRef.current || current), referenceMode: 'text',
      perspective: current.perspective, referenceImage: undefined,
      imageContinuationPrompt: current.imageContinuationPrompt })
  }

  useEffect(() => () => { ++imageRequestRef.current }, [])

  useEffect(() => {
    if (!moreMenuOpen) return
    const closeOnPointerDown = (event: PointerEvent) => {
      if (
        moreMenuRef.current
        && !moreMenuRef.current.contains(event.target as Node)
      ) {
        setMoreMenuOpen(false)
      }
    }
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setMoreMenuOpen(false)
    }
    document.addEventListener('pointerdown', closeOnPointerDown)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('pointerdown', closeOnPointerDown)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [moreMenuOpen])

  const status = backendConfig
    ? backendConfig.worker_online === false
        ? 'GPU worker offline'
        : `${backendConfig.rollout_type.toUpperCase()} · ${backendConfig.fps} FPS`
    : backendError
      ? 'Backend unavailable'
      : 'Connecting to backend'
  const setupNotice = backendError
    ? `Backend unavailable: ${backendError}`
    : sessionNotice
      ? sessionNotice
    : backendConfig?.worker_online === false
          ? 'The interface is ready. Waiting for the GPU worker.'
          : undefined

  return (
    <main className="setup-shell world-chat-setup">
      <header className="setup-header">
        <a className="brand" href="./" aria-label="Hy-Nomas home">
          <img
            className="brand-mark"
            src={`${import.meta.env.BASE_URL}hunyuan-logo.png`}
            alt=""
          />
          <span>Hy-Nomas</span>
        </a>
        <div className="setup-live-status">
          <div className="build-label">
            <span className="status-dot" />
            {status}
          </div>
          <ResourceStatus config={backendConfig} dark={theme === 'dark'} />
          <div className="more-options setup-more-options" ref={moreMenuRef}>
            <button
              className="icon-button"
              type="button"
              aria-label="More options"
              aria-controls="setup-options-menu"
              aria-expanded={moreMenuOpen}
              aria-haspopup="menu"
              onClick={() => setMoreMenuOpen((open) => !open)}
            >
              <MoreHorizontal size={19} />
            </button>
            {moreMenuOpen && (
              <div
                className="more-options-menu"
                id="setup-options-menu"
                role="menu"
                aria-label="Setup options"
              >
                <label className="more-options-setting">
                  <Bot size={16} />
                  <span><strong>Prompt enhancer</strong></span>
                  <select
                    aria-label="Prompt enhancer model"
                    value={promptEnhancerModel}
                    disabled={!backendConfig?.prompt_enhancement_enabled}
                    onChange={(event) => onPromptEnhancerModelChange(event.target.value)}
                  >
                    {(backendConfig?.prompt_enhancer_options ?? []).map((option) => (
                      <option key={option.id} value={option.id}>{option.label}</option>
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
                    disabled={!backendConfig?.prompt_enhancement_enabled}
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
                <label className="more-options-setting">
                  <Orbit size={16} />
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
                <label className="more-options-setting">
                  <Bot size={16} />
                  <span>
                    <strong>Thinking mode</strong>
                    <small>Prompt reasoning depth</small>
                  </span>
                  <select
                    aria-label="Thinking mode"
                    value={promptEnhancerReasoningEffort}
                    disabled={!backendConfig?.prompt_enhancement_enabled}
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
              </div>
            )}
          </div>
          <ThemeToggle theme={theme} onToggle={onThemeToggle} />
        </div>
      </header>

      <section className="world-chat-hero">
        <form className="world-chat-composer" onSubmit={submit}>
          <div className="world-chat-input-card">
            <div
              className={[
                'world-description-grid',
                isOneShot || imageControlsLocked ? 'single' : '',
              ].filter(Boolean).join(' ')}
            >
              {imageControlsLocked && (
                <div className="world-chat-field primary">
                  <label htmlFor="world-image-continuation">User prompt</label>
                  <textarea
                    id="world-image-continuation"
                    rows={3}
                    value={config.imageContinuationPrompt ?? ''}
                    onChange={(event) => update('imageContinuationPrompt', event.target.value)}
                    placeholder="Describe what should happen after the image reference, e.g. The subject turns and walks toward the window."
                    aria-describedby="world-image-continuation-hint"
                    required
                  />
                  <small id="world-image-continuation-hint">
                    The video starts with a 0.5-second reference image (T0), followed by your prompt (T1).
                  </small>
                </div>
              )}
              {!isOneShot && !imageControlsLocked && (
                <div className="world-chat-field environment">
                  <label htmlFor="world-environment">
                    Environment Description
                  </label>
                  <textarea
                    id="world-environment"
                    aria-label="Environment Description"
                    rows={2}
                    value={config.environmentDescription}
                    disabled={imageControlsLocked}
                    onChange={(event) =>
                      update('environmentDescription', event.target.value)
                    }
                    placeholder="e.g. A glass greenhouse drifts above a stormy ocean at blue hour"
                    required
                  />
                  <div
                    className="world-template-options"
                    aria-label="Environment description templates"
                  >
                    <span>Try a setting</span>
                    <div>
                      {environmentTemplates.map((template) => (
                        <button
                          type="button"
                          key={template.label}
                          title={template.value}
                          disabled={imageControlsLocked}
                          onClick={() =>
                            update('environmentDescription', template.value)
                          }
                        >
                          {template.label}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {!imageControlsLocked && <div className="world-chat-field primary">
                <label htmlFor="world-subject">
                  {isOneShot ? 'Prompt file' : 'Subject Description'}
                </label>
                <textarea
                  id="world-subject"
                  aria-label={isOneShot ? 'Prompt file' : 'Subject Description'}
                  rows={2}
                  value={config.subjectDescription}
                  disabled={imageControlsLocked}
                  onChange={(event) =>
                    update('subjectDescription', event.target.value)
                  }
                  placeholder={
                    isOneShot
                      ? '/path/to/prompt.txt'
                      : 'e.g. A lone astronaut in a weathered white pressure suit'
                  }
                  required
                />
                {!isOneShot && (
                  <div
                    className="world-template-options"
                    aria-label="Subject description templates"
                  >
                    <span>Try a subject</span>
                    <div>
                      {subjectTemplates.map((template) => (
                        <button
                          type="button"
                          key={template.label}
                          title={template.value}
                          disabled={imageControlsLocked}
                          onClick={() =>
                            update('subjectDescription', template.value)
                          }
                        >
                          {template.label}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>}
            </div>

            {imageControlsLocked && (
              <div className="reference-image-control">
                <small>Nomas-PE captions your image using your selected perspective. Enter world creates the image reference, then generates your second segment.</small>
                <small>The full image is fitted to 1280 × 720 with padding; nothing is cropped.</small>
                {backendConfig?.reference_image_enabled === false && (
                  <small>Image references are unavailable on this worker.</small>
                )}
                {backendConfig?.image_captioning_enabled === false && <small>Image captioning is unavailable.</small>}
                {imageUploading && <p role="status">Nomas-PE is captioning your image…</p>}

                {config.referenceImage && (
                  <div className="reference-image-preview">
                    <img src={config.referenceImage.url} alt="World reference" />

                  </div>
                )}
                <button type="button" onClick={removeImage}>{imageUploading ? 'Cancel upload' : 'Remove image'}</button>
              </div>
            )}
            {imageError && <p role="alert">{imageError}</p>}

            <div className="world-chat-meta-row">
              <div className="world-chat-controls perspective-upload-row">
                <label
                  className="world-chat-perspective-select"
                  htmlFor="world-perspective"
                >
                  <span>Perspective</span>
                  <select
                    id="world-perspective"
                    aria-label="Perspective"
                    value={config.perspective}
                    onChange={(event) =>
                      update(
                        'perspective',
                        event.target.value as Perspective,
                      )
                    }
                  >
                    {perspectives.map(({ value, label }) => (
                      <option
                        key={value}
                        value={value}
                      >
                        {label}
                      </option>
                    ))}
                  </select>
                </label>
                {!isOneShot && <div className="image-upload-section">
                  <span>Image reference</span>
                  <button className="reference-mode-toggle" type="button"
                    disabled={imageUploading || backendConfig?.reference_image_enabled === false || backendConfig?.image_captioning_enabled === false}
                    onClick={() => uploadInputRef.current?.click()}>
                    <Images size={18} aria-hidden="true" />
                    <span>Upload image</span>
                  </button>
                  <input ref={uploadInputRef} hidden type="file" aria-label="Upload image file"
                    accept="image/jpeg,image/jpg,image/pjpeg,image/png,image/webp,.jpg,.jpeg,.jfif,.png,.webp"
                    onChange={(event) => {
                      void uploadImage(event.target.files?.[0], true)
                      event.target.value = ''
                    }} />
                </div>}
              </div>

            </div>

            <footer className="world-chat-footer">
              {setupNotice && <p>{setupNotice}</p>}
              <div>
                <button
                  className="primary-button"
                  type="submit"
                  disabled={
                    backendConfig?.worker_online === false
                    || imageUploading
                    || !captionReady
                    || !setupInputReady
                  }
                >
                  Enter world
                  <ArrowRight size={17} />
                </button>
              </div>
            </footer>
          </div>
        </form>
      </section>

      <WorldGallery
        savedSamples={backendConfig?.gallery_items}
        hiddenSampleIds={backendConfig?.gallery_hidden_ids}
        onSelect={selectGallerySample}
      />
    </main>
  )
}

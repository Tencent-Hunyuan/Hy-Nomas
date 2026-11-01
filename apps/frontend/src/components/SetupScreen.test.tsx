import '@testing-library/jest-dom/vitest'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import type { WorldConfig } from '../types'
import { afterEach, expect, it, vi } from 'vitest'
import { SetupScreen } from './SetupScreen'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

it.each([
  ['cat.png', 'image/png'],
  ['photo.jpeg', 'image/jpeg'],
  ['photo.jpg', 'image/jpg'],
  ['photo.JPG', ''],
])('uploads %s (%s) and lets the user remove it', async (name, type) => {
  const user = userEvent.setup()
  const referenceImage = { id: 'a'.repeat(32), url: './outputs/reference_images/a.jpg', width: 64, height: 64, caption_status: 'ready' as const }
  const caption = { caption_status: 'ready', perspective: 'third-person observer', subject_description: 'A red cube', environment_description: 'A wooden table' }
  let finishCaption!: (response: Response) => void
  const captionResponse = new Promise<Response>((resolve) => { finishCaption = resolve })
  const fetch = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ ...referenceImage, caption_status: 'captioning' }), { status: 201 }))
    .mockReturnValueOnce(captionResponse)
  vi.stubGlobal('fetch', fetch)
  const onChange = vi.fn()
  const props = {
    config: { perspective: 'first-person' as const, subjectDescription: 'A cat', environmentDescription: 'A room' },
    promptEnhancerModel: 'Nomas-PE', onPromptEnhancerModelChange: vi.fn(),
    promptEnhancerReasoningEffort: 'none' as const, promptEnhancerReasoningEffortOptions: [],
    onPromptEnhancerReasoningEffortChange: vi.fn(), promptEnhancerSeedMin: 0,
    promptEnhancerSeedMax: 2147483647, onPromptEnhancerSeedChange: vi.fn(),
    inferenceSteps: 8, inferenceStepsOptions: [8], onInferenceStepsChange: vi.fn(),
    onChange, onStart: vi.fn(), theme: 'dark' as const, onThemeToggle: vi.fn(),
  }
  function Setup() {
    const [config, setConfig] = useState<WorldConfig>(props.config)
    return <SetupScreen {...props} config={config} onChange={(next) => { onChange(next); setConfig(next) }} />
  }
  render(<Setup />)
  expect(screen.queryByRole('button', { name: 'Image reference mode' })).not.toBeInTheDocument()
  const uploadButton = screen.getByRole('button', { name: 'Upload image' })
  expect(uploadButton.closest('.world-chat-controls')).toContainElement(screen.getByLabelText('Perspective'))
  expect(screen.getByLabelText('Subject Description')).toHaveValue('A cat')
  await user.click(uploadButton)
  expect(screen.getByLabelText('Subject Description')).toHaveValue('A cat')
  const file = new File(['image'], name, { type })
  await user.upload(screen.getByLabelText('Upload image file'), file)
  expect(screen.queryByLabelText('Subject Description')).not.toBeInTheDocument()
  expect(screen.queryByLabelText('Environment Description')).not.toBeInTheDocument()
  expect(screen.getAllByRole('textbox')).toHaveLength(1)
  expect(screen.getByLabelText('Perspective')).toBeEnabled()
  expect(screen.getByRole('button', { name: 'Enter world' })).toBeDisabled()
  const instruction = 'The cube slides to the right.'
  await user.type(screen.getByLabelText('User prompt'), instruction)
  finishCaption(new Response(JSON.stringify(caption), { status: 200 }))
  const captionedConfig = { ...props.config, referenceMode: 'image', referenceImage, imageContinuationPrompt: instruction, subjectDescription: caption.subject_description, environmentDescription: caption.environment_description }
  await waitFor(() => expect(onChange).toHaveBeenCalledWith(captionedConfig))
  expect(fetch).toHaveBeenCalledWith(expect.stringContaining('v1/reference-images'), expect.objectContaining({ method: 'POST', body: file }))
  expect(fetch).toHaveBeenCalledWith(expect.stringContaining('perspective=first-person'), expect.anything())
  expect(screen.getByAltText('World reference')).toHaveAttribute('src', referenceImage.url)
  expect(screen.queryByLabelText('Subject Description')).not.toBeInTheDocument()
  expect(screen.queryByLabelText('Environment Description')).not.toBeInTheDocument()
  expect(screen.getByLabelText('User prompt')).toHaveValue(instruction)
  expect(screen.getByRole('button', { name: 'Enter world' })).toBeEnabled()
  await user.clear(screen.getByLabelText('User prompt'))
  expect(screen.getByRole('button', { name: 'Enter world' })).toBeDisabled()
  await user.type(screen.getByLabelText('User prompt'), instruction)
  expect(screen.getByLabelText('Perspective')).toBeEnabled()
  expect(screen.getByLabelText('Perspective')).toHaveValue('first-person')
  await user.selectOptions(screen.getByLabelText('Perspective'), 'second-person')
  expect(onChange).toHaveBeenLastCalledWith({ ...captionedConfig, perspective: 'second-person' })
  await user.click(screen.getByRole('button', { name: 'Remove image' }))
  expect(screen.getByLabelText('Subject Description')).toHaveValue('A cat')
  expect(screen.getByLabelText('Environment Description')).toHaveValue('A room')
  expect(screen.getByLabelText('Perspective')).toHaveValue('second-person')
  expect(screen.queryByLabelText('User prompt')).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Upload image' })).toBeEnabled()
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ referenceImage: undefined, referenceMode: 'text' }))

})

it('ignores a pending upload after switching back to text-only mode', async () => {
  const user = userEvent.setup()
  let finishUpload!: (response: Response) => void
  const upload = new Promise<Response>((resolve) => { finishUpload = resolve })
  const fetch = vi.fn().mockReturnValue(upload)
  vi.stubGlobal('fetch', fetch)
  const onChange = vi.fn()
  function Setup() {
    const [config, setConfig] = useState<WorldConfig>({
      perspective: 'first-person', subjectDescription: 'A cat', environmentDescription: 'A room',
    })
    return <SetupScreen config={config} onChange={(next) => { onChange(next); setConfig(next) }}
      onStart={vi.fn()} theme="dark" onThemeToggle={vi.fn()}
      promptEnhancerModel="kimi-k3" onPromptEnhancerModelChange={vi.fn()}
      promptEnhancerReasoningEffort="none" promptEnhancerReasoningEffortOptions={[]}
      onPromptEnhancerReasoningEffortChange={vi.fn()}
      promptEnhancerSeedMin={0} promptEnhancerSeedMax={2147483647} onPromptEnhancerSeedChange={vi.fn()}
      inferenceSteps={8} inferenceStepsOptions={[8]} onInferenceStepsChange={vi.fn()} />
  }
  render(<Setup />)
  await user.upload(screen.getByLabelText('Upload image file'), new File(['jpeg'], 'cat.jpg', { type: 'image/jpeg' }))
  await user.click(screen.getByRole('button', { name: 'Cancel upload' }))
  onChange.mockClear()
  await act(async () => finishUpload(new Response(JSON.stringify({ id: 'late-upload', caption_status: 'captioning' }), { status: 201 })))
  expect(onChange).not.toHaveBeenCalled()
  expect(fetch).toHaveBeenCalledTimes(1)
  expect(screen.getByLabelText('Subject Description')).toHaveValue('A cat')
  expect(screen.getByLabelText('Environment Description')).toHaveValue('A room')
  expect(screen.getByRole('button', { name: 'Enter world' })).toBeEnabled()
  expect(screen.queryByLabelText('Reference image')).not.toBeInTheDocument()
})

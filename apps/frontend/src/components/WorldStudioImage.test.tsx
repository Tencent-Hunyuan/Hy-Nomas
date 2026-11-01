import '@testing-library/jest-dom/vitest'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { WorldStudio } from './WorldStudio'

afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks() })

const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), {
  status, headers: { 'Content-Type': 'application/json' },
})

it.each([false, true])('waits for the image segment and handles failure=%s before sending segment 2', async (fail) => {
  vi.spyOn(console, 'log').mockImplementation(() => {})
  vi.spyOn(console, 'groupCollapsed').mockImplementation(() => {})
  vi.spyOn(console, 'groupEnd').mockImplementation(() => {})
  const posted: Array<Record<string, any>> = []
  let finishReference!: (response: Response) => void
  const referenceStatus = new Promise<Response>((resolve) => { finishReference = resolve })
  vi.stubGlobal('fetch', vi.fn(async (url: string, options?: RequestInit) => {
    if (String(url).endsWith('v1/generate') && options?.method === 'POST') {
      posted.push(JSON.parse(options.body as string))
      if (posted.length === 1) return json({ id: 'reference-job', status: 'queued' }, 202)
      return json({ id: 'continuation', video_url: 'outputs/continued.mp4', frames: 132,
        segment_frames: 120, fps: 24, generation_seconds: 1, prompt_enhanced: true })
    }
    if (String(url).endsWith('v1/generate/reference-job')) return referenceStatus
    return json({ status: 'active' })
  }))
  const instruction = 'The cube slides toward the window.'
  render(<WorldStudio
    config={{ perspective: 'second-person', subjectDescription: 'A red cube',
      environmentDescription: 'A sunlit room', imageContinuationPrompt: instruction,
      referenceImage: { id: 'image-id', url: 'image.jpg', width: 96, height: 64, caption_status: 'ready' } }}
    backendConfig={{ rollout_type: 'seg_ar', prompt_source: 'prompt', fps: 24,
      num_frames: 120, num_frames_source: 'request_or_config', worker_online: true,
      prompt_enhancement_enabled: true }}
    promptEnhancerModel="kimi-k3"
    onReconfigure={vi.fn()} onSessionExpired={vi.fn()} theme="dark" onThemeToggle={vi.fn()}
  />)
  await waitFor(() => expect(posted).toHaveLength(1))
  expect(screen.getAllByText('Loading image reference into KV cache…').length).toBeGreaterThan(0)
  expect(posted[0].reference_image_id).toBe('image-id')
  expect(posted[0].reset_session).toBe(true)
  expect(posted[0].pe_input.perspective).toBe('second-person')
  expect(posted[0].pe_input).not.toHaveProperty('action')
  expect(JSON.stringify(posted[0])).not.toContain(instruction)
  await act(async () => finishReference(fail
    ? json({ id: 'reference-job', status: 'failed', error: 'Encoder failed' })
    : json({ id: 'reference-job', status: 'succeeded', result: {
      id: 'reference', video_url: 'outputs/reference.mp4', frames: 12,
      segment_frames: 12, fps: 24, generation_seconds: 1, prompt_enhanced: true,
    } })))
  if (fail) {
    await waitFor(() => expect(screen.getAllByText(/Encoder failed/).length).toBeGreaterThan(0))
    expect(posted).toHaveLength(1)
  } else {
    await waitFor(() => expect(posted).toHaveLength(2))
    expect(posted[1]).not.toHaveProperty('reference_image_id')
    expect(posted[1].reset_session).toBe(false)
    expect(posted[1].session_id).toBe(posted[0].session_id)
    expect(posted[1].pe_input).toMatchObject({ action: instruction, prompt_enhancer_model: 'kimi-k3' })
    await waitFor(() => expect(screen.getByLabelText('Direct the world')).toBeEnabled())
    expect(posted).toHaveLength(2)
    expect(screen.getAllByText('T0-Reference Image').length).toBeGreaterThan(0)
    expect(screen.getAllByText('T1 · TEXT').length).toBeGreaterThan(0)
  }
})

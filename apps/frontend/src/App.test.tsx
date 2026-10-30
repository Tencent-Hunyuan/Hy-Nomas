import '@testing-library/jest-dom/vitest'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { normalizePerspective } from './perspective'
import { GALLERY_ITEM_REMOVED_EVENT } from './types'

afterEach(cleanup)

beforeEach(() => {
  window.localStorage.clear()
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          rollout_type: 'seg_ar',
          prompt_source: 'prompt',
          num_frames: 49,
          num_frames_source: 'request_or_config',
          fps: 24,
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        },
      ),
    ),
  )
})

async function enterWorld(
  subject = 'A silver fox exploring a floating greenhouse',
  environment = 'A floating greenhouse filled with silver-leaf plants',
) {
  const user = userEvent.setup()
  await user.type(screen.getByLabelText('Subject Description'), subject)
  await user.type(
    screen.getByLabelText('Environment Description'),
    environment,
  )
  await user.click(screen.getByRole('button', { name: 'Enter world' }))
  return user
}

function jsonResponse(value: unknown, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function mockMultiHostFetch(
  loadAllocations: (options?: RequestInit) => Promise<Response>,
) {
  const dispatchSession = {
    user_id: 'k1',
    status: 'assigned',
    host_id: 'sp16-00',
    waitlist_size: 0,
    lease_seconds: 900,
    lease_expires_at: Date.now() / 1000 + 900,
    server_time: Date.now() / 1000,
  }
  vi.mocked(fetch).mockImplementation(async (url, options) => {
    const path = String(url)
    if (path.endsWith('auth/allocations')) {
      return loadAllocations(options)
    }
    if (
      path.endsWith('auth/reconnect')
      || path.endsWith('auth/status')
    ) {
      return jsonResponse(dispatchSession)
    }
    if (path.endsWith('health')) {
      return jsonResponse({
        hosts: [{ online: true, available: false }],
        active_users: 1,
        waitlist_size: 0,
      })
    }
    if (path.endsWith('v1/config')) {
      return jsonResponse({
        rollout_type: 'seg_ar',
        prompt_source: 'prompt',
        num_frames: 49,
        num_frames_source: 'request_or_config',
        fps: 24,
        worker_online: true,
      })
    }
    return jsonResponse({ status: 'ok' })
  })
}

describe('Hy-Nomas', () => {
  it('returns to setup when the worker releases an idle world despite browser activity', async () => {
    let sessionStatus = 'active'
    let statusUnavailable = false
    let statusChecks = 0
    vi.mocked(fetch).mockImplementation(async (url) => {
      const path = String(url)
      if (path.endsWith('v1/config')) {
        return jsonResponse({
          rollout_type: 'seg_ar', prompt_source: 'prompt',
          num_frames: 49, num_frames_source: 'request_or_config', fps: 24,
          worker_online: true, prompt_enhancement_enabled: true,
        })
      }
      if (path.endsWith('v1/generate')) {
        return jsonResponse({
          id: 'idle-world', status: 'succeeded', video_url: 'outputs/idle.mp4',
          frames: 120, fps: 24, generation_seconds: 1, prompt_enhanced: true,
        })
      }
      if (/v1\/sessions\/[^/]+\/status$/.test(path)) {
        statusChecks += 1
        return statusUnavailable
          ? jsonResponse({ detail: 'Unavailable' }, 503)
          : jsonResponse({ status: sessionStatus })
      }
      return jsonResponse({ status: 'ok' })
    })

    render(<App />)
    await enterWorld()
    await vi.waitFor(() => expect(statusChecks).toBe(1))
    expect(screen.queryByRole('button', { name: 'Enter world' })).not.toBeInTheDocument()

    statusUnavailable = true
    fireEvent.focus(window)
    await vi.waitFor(() => expect(statusChecks).toBe(2))
    expect(screen.queryByRole('button', { name: 'Enter world' })).not.toBeInTheDocument()

    statusUnavailable = false
    sessionStatus = 'expired'
    fireEvent.pointerDown(window)
    await screen.findByRole('button', { name: 'Enter world' }, { timeout: 6_000 })
    expect(screen.getByText('Your world session has ended. Create a new world to continue.')).toBeInTheDocument()
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith('/release'))).toBe(true)
  }, 10_000)

  it('retries transient active-allocation fetch failures before showing data', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    let allocationAttempts = 0
    mockMultiHostFetch(async () => {
      allocationAttempts += 1
      if (allocationAttempts < 3) {
        throw new TypeError('Failed to fetch')
      }
      return jsonResponse({
        allocations: [{ user_id: 'alice', host_id: 'sp16-01' }],
        count: 1,
        server_time: Date.now() / 1000,
      })
    })

    render(<App multiHost />)
    fireEvent.click(
      await screen.findByRole('button', { name: 'Active users' }),
    )
    await vi.advanceTimersByTimeAsync(1_000)

    await vi.waitFor(() => {
      expect(allocationAttempts).toBe(3)
      expect(screen.getByText('alice')).toBeInTheDocument()
    })
    expect(screen.getByText('1 active allocation')).toBeInTheDocument()
    expect(screen.queryByText('Failed to fetch')).not.toBeInTheDocument()
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('auth/reconnect'),
      expect.objectContaining({
        headers: { 'X-Nomas-Page-Id': expect.any(String) },
      }),
    )
    vi.useRealTimers()
  })

  it('shows active users in the compact SP16 status instead of free slots', async () => {
    vi.mocked(fetch).mockImplementation(async (url) => {
      const path = String(url)
      if (
        path.endsWith('auth/reconnect')
        || path.endsWith('auth/status')
      ) {
        return jsonResponse({
          user_id: 'alice',
          status: 'assigned',
          host_id: 'sp16-00',
          waitlist_size: 0,
          lease_seconds: 900,
          lease_expires_at: Date.now() / 1000 + 900,
          server_time: Date.now() / 1000,
        })
      }
      if (path.endsWith('health')) {
        return jsonResponse({
          hosts: Array.from({ length: 8 }, (_, index) => ({
            id: `sp16-0${index}`,
            online: index < 6,
            available: false,
            busy: true,
          })),
          active_users: 6,
          waitlist_size: 0,
        })
      }
      if (path.endsWith('v1/config')) {
        return jsonResponse({
          rollout_type: 'seg_ar',
          prompt_source: 'prompt',
          num_frames: 49,
          num_frames_source: 'request_or_config',
          fps: 24,
          worker_online: true,
        })
      }
      return jsonResponse({ status: 'ok' })
    })

    render(<App multiHost />)

    expect(
      await screen.findByText('6 / 8 active users'),
    ).toBeInTheDocument()
    expect(screen.queryByText('0 / 8 SP16 slots available')).not
      .toBeInTheDocument()
  })

  it('keeps existing allocations after retry exhaustion and clears the error on recovery', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    let phase: 'initial' | 'failing' | 'recovered' = 'initial'
    let failingAttempts = 0
    mockMultiHostFetch(async () => {
      if (phase === 'failing') {
        failingAttempts += 1
        throw new TypeError('Failed to fetch')
      }
      const recovered = phase === 'recovered'
      return jsonResponse({
        allocations: [{
          user_id: recovered ? 'bob' : 'alice',
          host_id: recovered ? 'sp16-02' : 'sp16-01',
        }],
        count: 1,
        server_time: Date.now() / 1000,
      })
    })

    render(<App multiHost />)
    fireEvent.click(
      await screen.findByRole('button', { name: 'Active users' }),
    )
    expect(await screen.findByText('alice')).toBeInTheDocument()

    phase = 'failing'
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }))
    await vi.advanceTimersByTimeAsync(5_000)

    await vi.waitFor(() => {
      expect(failingAttempts).toBe(5)
      expect(screen.getByText('Failed to fetch')).toBeInTheDocument()
    })
    expect(screen.getByText('alice')).toBeInTheDocument()
    expect(screen.getByText('sp16-01')).toBeInTheDocument()

    phase = 'recovered'
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }))
    await vi.waitFor(() => {
      expect(screen.getByText('bob')).toBeInTheDocument()
      expect(screen.queryByText('Failed to fetch')).not.toBeInTheDocument()
    })
    expect(screen.queryByText('alice')).not.toBeInTheDocument()
    vi.useRealTimers()
  })

  it('aborts an active-allocation request when the app unmounts', async () => {
    let allocationSignal: AbortSignal | undefined
    mockMultiHostFetch((options) => new Promise<Response>((_resolve, reject) => {
      allocationSignal = options?.signal ?? undefined
      allocationSignal?.addEventListener('abort', () => {
        reject(new DOMException('The operation was aborted.', 'AbortError'))
      }, { once: true })
    }))

    const rendered = render(<App multiHost />)
    fireEvent.click(
      await screen.findByRole('button', { name: 'Active users' }),
    )
    await vi.waitFor(() => expect(allocationSignal).toBeDefined())

    rendered.unmount()
    expect(allocationSignal?.aborted).toBe(true)
  })

  it('uses dark mode by default and remembers the light-mode choice', async () => {
    const user = userEvent.setup()
    const rendered = render(<App />)

    expect(document.documentElement.dataset.theme).toBe('dark')
    await user.click(
      screen.getByRole('button', { name: 'Switch to light mode' }),
    )
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(window.localStorage.getItem('nomas-world-engine-theme')).toBe('light')

    rendered.unmount()
    render(<App />)
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('v1/config'),
      expect.objectContaining({
        headers: {
          Accept: 'application/json',
          'X-Nomas-Client-Id': expect.any(String),
        },
      }),
    )
  })

  it('fills subject and environment descriptions from quick templates', async () => {
    const user = userEvent.setup()
    render(<App />)

    expect(
      screen.getAllByRole('textbox').slice(0, 2).map((field) =>
        field.getAttribute('aria-label'),
      ),
    ).toEqual(['Environment Description', 'Subject Description'])

    await user.click(screen.getByRole('button', { name: 'Clockwork fox' }))
    await user.click(screen.getByRole('button', { name: 'Mossy observatory' }))

    expect(screen.getByLabelText('Subject Description')).toHaveValue(
      'A small brass clockwork fox with green glass eyes and a red scarf',
    )
    expect(screen.getByLabelText('Environment Description')).toHaveValue(
      'A moss-covered observatory at dawn, with tall windows and a rotating brass orrery.',
    )
    expect(screen.queryByText('World Simulator')).not.toBeInTheDocument()
    expect(
      screen.queryByText(/Multiple isolated GPU slots/),
    ).not.toBeInTheDocument()
  })

  it('uses the configured model without a checkpoint switch control', async () => {
    vi.mocked(fetch).mockResolvedValue(jsonResponse({
      rollout_type: 'seg_ar',
      prompt_source: 'prompt',
      num_frames: 49,
      num_frames_source: 'request_or_config',
      fps: 24,
      worker_online: true,
      worker_status: 'ready',
      model_name: 'inference',
      dit_weight: '/models/configured.safetensors',
      num_inference_steps: 12,
      default_num_inference_steps: 12,
      num_inference_steps_options: [4, 6, 8, 12],
    }))
    render(<App />)
    await userEvent.setup().click(
      screen.getByRole('button', { name: 'More options' }),
    )
    await vi.waitFor(() => {
      expect(screen.getByLabelText('Inference steps')).toHaveValue('12')
    })
    expect(screen.queryByLabelText('Inference model')).not.toBeInTheDocument()
    expect(vi.mocked(fetch).mock.calls.some(([url]) =>
      String(url).endsWith('v1/model'),
    )).toBe(false)
  })

  it('keeps internal dry-run controls out of the setup experience', () => {
    render(<App />)
    expect(
      screen.queryByRole('button', { name: 'Explore sample' }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByText('Start from a world that already feels alive.'),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByText(/Hover to read the direction/),
    ).not.toBeInTheDocument()
    expect(screen.getByText('World gallery')).toBeInTheDocument()
    expect(screen.getByRole('region', {
      name: 'Selected generated worlds',
    })).toBeInTheDocument()
  })

  it('keeps the app gallery list in sync after a case is hidden', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({
        rollout_type: 'seg_ar',
        prompt_source: 'prompt',
        num_frames: 49,
        num_frames_source: 'request_or_config',
        fps: 24,
        gallery_items: [{
          id: 'saved-gallery-case',
          title: 'Saved Gallery Case',
          video: 'gallery/favorites/saved-gallery-case.mp4',
          duration: '8s',
          perspective: 'first-person',
          subject: 'An explorer',
          attributes: 'A crystal canyon',
          direction: 'Walk forward',
        }],
      }),
    )
    render(<App />)

    expect(
      await screen.findAllByLabelText('Gallery video Saved Gallery Case'),
    ).toHaveLength(2)
    window.dispatchEvent(
      new CustomEvent(GALLERY_ITEM_REMOVED_EVENT, {
        detail: { itemId: 'saved-gallery-case' },
      }),
    )
    expect(GALLERY_ITEM_REMOVED_EVENT).toBe('nomas:gallery-item-removed')

    await vi.waitFor(() => {
      expect(
        screen.queryByLabelText('Gallery video Saved Gallery Case'),
      ).not.toBeInTheDocument()
    })
  })

  it('requires subject and environment descriptions before entering the world', async () => {
    const user = userEvent.setup()
    render(<App />)

    const enterButton = screen.getByRole('button', { name: 'Enter world' })
    expect(enterButton).toBeDisabled()

    expect(screen.queryByLabelText('Style')).not.toBeInTheDocument()
    const perspective = screen.getByLabelText('Perspective')
    expect(perspective).toHaveValue('first-person')
    expect(within(perspective).getAllByRole('option')).toHaveLength(4)
    await user.selectOptions(perspective, 'third-person follow')
    expect(perspective).toHaveAttribute(
      'title',
      'Camera follows behind or alongside the main subject.',
    )
    expect(screen.getByText(
      'Camera follows behind or alongside the main subject.',
    )).toBeInTheDocument()
    await user.type(
      screen.getByLabelText('Subject Description'),
      'A paper dragon with red-and-gold folded wings',
    )
    expect(enterButton).toBeDisabled()
    await user.type(
      screen.getByLabelText('Environment Description'),
      'An animated seaside city and harbor at sunset',
    )
    await user.click(enterButton)

    expect(screen.getByLabelText('World model output')).toBeInTheDocument()
    expect(screen.getByText('Third person')).toBeInTheDocument()
    expect(
      screen.getByText('A paper dragon with red-and-gold folded wings'),
    ).toBeInTheDocument()
    expect(screen.queryByText('Ready for direction')).not.toBeInTheDocument()
    expect(
      screen.queryByRole('button', { name: 'Close world agent' }),
    ).not.toBeInTheDocument()
  })

  it('opens the Studio options menu and closes it with Escape', async () => {
    render(<App />)
    const user = await enterWorld()

    const trigger = screen.getByRole('button', { name: 'More options' })
    await user.click(trigger)
    expect(trigger).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByRole('menu', { name: 'World options' })).toBeInTheDocument()
    expect(screen.getByRole('menuitem', { name: /Download video/ })).toBeDisabled()
    expect(screen.getByRole('menuitem', { name: /Share video/ })).toBeDisabled()
    expect(
      screen.queryByRole('menuitem', { name: /New world/ }),
    ).not.toBeInTheDocument()

    await user.keyboard('{Escape}')
    expect(screen.queryByRole('menu', { name: 'World options' })).not.toBeInTheDocument()
    expect(trigger).toHaveAttribute('aria-expanded', 'false')
  })

  it('accepts multiple prompts in the agent panel', async () => {
    render(<App />)
    const user = await enterWorld()
    const composer = screen.getByLabelText('Direct the world')

    await user.type(composer, 'Make it begin to rain{Enter}')
    expect(screen.getByText('Make it begin to rain')).toBeInTheDocument()
    expect(screen.getByText('PROMPT 1')).toBeInTheDocument()

    await user.type(composer, 'The fox pauses under a red umbrella{Enter}')
    expect(screen.getByText('The fox pauses under a red umbrella')).toBeInTheDocument()
    expect(screen.getByText('PROMPT 2')).toBeInTheDocument()
  })

  it('sends natural-language actions through prompt enhancement', async () => {
    const generateResponse = new Response(
      JSON.stringify({
        id: 'gen_pe',
        status: 'succeeded',
        video_url: 'outputs/pe.mp4',
        frames: 120,
        fps: 24,
        generation_seconds: 1,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      },
    )
    vi.mocked(fetch).mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          rollout_type: 'seg_ar',
          prompt_source: 'prompt',
          num_frames: 49,
          num_frames_source: 'request_or_config',
          fps: 24,
          prompt_enhancement_enabled: true,
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        },
      ),
    ).mockResolvedValueOnce(generateResponse)

    render(<App />)
    const user = await enterWorld('An astronaut in a white pressure suit')
    await user.type(screen.getByLabelText('Direct the world'), 'Walk forward{Enter}')

    const lastCall = vi.mocked(fetch).mock.calls.at(-1)
    const body = JSON.parse(
      (lastCall?.[1] as RequestInit).body as string,
    ) as {
      pe_input: {
        perspective: string
        subject_description: string
        environment_description: string
        session_id: string
      }
    }
    expect(body.pe_input).toMatchObject({
      perspective: 'first-person',
      subject_description: 'An astronaut in a white pressure suit',
      environment_description:
        'A floating greenhouse filled with silver-leaf plants',
      prompt_enhancer_model: 'Nomas-PE',
      prompt_enhancer_reasoning_effort: 'none',
    })
    expect(body).toMatchObject({ num_inference_steps: 8 })
    expect(body.pe_input).not.toHaveProperty('style')
    expect(body.pe_input.session_id).toBeTruthy()
  })

  it('keeps the single-model selector and replaces an unavailable stored model', async () => {
    window.localStorage.setItem('nomas-prompt-enhancer-model', 'removed-model')
    vi.mocked(fetch)
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            rollout_type: 'seg_ar',
            prompt_source: 'prompt',
            num_frames: 49,
            num_frames_source: 'request_or_config',
            fps: 24,
            worker_online: true,
            prompt_enhancement_enabled: true,
            default_prompt_enhancer_model: 'kimi-k3',
            prompt_enhancer_options: [
              { id: 'kimi-k3', label: 'Kimi-K3', default: true },
            ],
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )
      .mockResolvedValue(
        new Response(
          JSON.stringify({
            id: 'gen_pe',
            status: 'succeeded',
            video_url: 'outputs/pe.mp4',
            frames: 120,
            fps: 24,
            generation_seconds: 1,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )

    render(<App />)
    const setupUser = userEvent.setup()
    await setupUser.click(screen.getByRole('button', { name: 'More options' }))
    const setupSelector = screen.getByLabelText('Prompt enhancer model')
    expect(setupSelector).toHaveValue('kimi-k3')
    expect(within(setupSelector).getAllByRole('option')).toHaveLength(1)
    await setupUser.click(screen.getByRole('button', { name: 'More options' }))
    const user = await enterWorld()
    expect(
      screen.queryByLabelText('Prompt enhancer model'),
    ).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'More options' }))
    const selector = screen.getByLabelText('Prompt enhancer model')
    expect(selector).toHaveValue('kimi-k3')
    expect(within(selector).getAllByRole('option')).toHaveLength(1)
    expect(screen.getByLabelText('Thinking mode')).toBeEnabled()
    expect(screen.getByLabelText('LLM random seed')).toBeEnabled()
    expect(window.localStorage.getItem('nomas-prompt-enhancer-model')).toBe('kimi-k3')

    await vi.waitFor(() => {
      expect(
        vi.mocked(fetch).mock.calls.some(
          ([url]) => url === `${import.meta.env.BASE_URL}v1/generate`,
        ),
      ).toBe(true)
    })
    const generateCall = vi.mocked(fetch).mock.calls.find(
      ([url]) => url === `${import.meta.env.BASE_URL}v1/generate`,
    )
    expect(generateCall).toBeDefined()
    const body = JSON.parse(
      (generateCall?.[1] as RequestInit).body as string,
    ) as {
      pe_input: {
        prompt_enhancer_model: string
        prompt_enhancer_reasoning_effort: string
      }
    }
    expect(body.pe_input.prompt_enhancer_model).toBe('kimi-k3')
    expect(body.pe_input.prompt_enhancer_reasoning_effort).toBe('none')
  })

  it('restores a stored model only when it is advertised by the API', async () => {
    window.localStorage.setItem('nomas-prompt-enhancer-model', 'org/Model-A')
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({
      rollout_type: 'seg_ar',
      prompt_source: 'prompt',
      num_frames: 49,
      num_frames_source: 'request_or_config',
      fps: 24,
      worker_online: true,
      prompt_enhancement_enabled: true,
      default_prompt_enhancer_model: 'kimi-k3',
      prompt_enhancer_options: [
        { id: 'kimi-k3', label: 'Kimi-K3', default: true },
        { id: 'org/Model-A', label: 'org/Model-A' },
      ],
    }))
    render(<App />)
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'More options' }))
    expect(screen.getByLabelText('Prompt enhancer model')).toHaveValue('org/Model-A')
    expect(window.localStorage.getItem('nomas-prompt-enhancer-model')).toBe('org/Model-A')
  })

  it('uses the backend default thinking mode when no preference is stored', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          rollout_type: 'seg_ar',
          prompt_source: 'prompt',
          num_frames: 49,
          num_frames_source: 'request_or_config',
          fps: 24,
          worker_online: true,
          prompt_enhancement_enabled: true,
          prompt_enhancer_reasoning_effort_options: [
            'low',
            'medium',
            'high',
          ],
          default_prompt_enhancer_reasoning_effort: 'medium',
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        },
      ),
    )

    render(<App />)
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'More options' }))

    const selector = screen.getByLabelText('Thinking mode')
    expect(selector).toHaveValue('medium')
    expect(within(selector).getAllByRole('option')).toHaveLength(3)
    expect(window.localStorage.getItem(
      'nomas-prompt-enhancer-reasoning-effort',
    )).toBe('medium')
  })

  it('persists and sends an optional LLM random seed on every PE turn', async () => {
    window.localStorage.setItem('nomas-prompt-enhancer-seed', '424242')
    let generated = 0
    vi.mocked(fetch).mockImplementation(async (url) => {
      const path = String(url)
      if (path.endsWith('v1/config')) {
        return jsonResponse({
          rollout_type: 'seg_ar',
          prompt_source: 'prompt',
          num_frames: 49,
          num_frames_source: 'request_or_config',
          fps: 24,
          worker_online: true,
          prompt_enhancement_enabled: true,
          prompt_enhancer_seed_min: 0,
          prompt_enhancer_seed_max: 2147483647,
        })
      }
      if (path.endsWith('v1/generate')) {
        generated += 1
        return jsonResponse({
          id: `gen-seed-${generated}`,
          status: 'succeeded',
          video_url: `outputs/seed-${generated}.mp4`,
          frames: generated * 120,
          segment_frames: 120,
          fps: 24,
          generation_seconds: 1,
        })
      }
      return jsonResponse({ status: 'ok' })
    })

    render(<App />)
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'More options' }))
    const setupSeed = screen.getByLabelText('LLM random seed')
    expect(setupSeed).toHaveValue(424242)
    await user.clear(setupSeed)
    await user.type(setupSeed, '123456')
    expect(window.localStorage.getItem('nomas-prompt-enhancer-seed')).toBe(
      '123456',
    )
    await user.click(screen.getByRole('button', { name: 'More options' }))

    await user.type(screen.getByLabelText('Subject Description'), 'A red kite')
    await user.type(
      screen.getByLabelText('Environment Description'),
      'A windy coastal meadow',
    )
    await user.click(screen.getByRole('button', { name: 'Enter world' }))
    await screen.findByText('120 frames')

    let generateCalls = vi.mocked(fetch).mock.calls.filter(
      ([url]) => String(url).endsWith('v1/generate'),
    )
    let body = JSON.parse(
      (generateCalls[0][1] as RequestInit).body as string,
    ) as { pe_input: { prompt_enhancer_seed?: number } }
    expect(body.pe_input.prompt_enhancer_seed).toBe(123456)

    await user.click(screen.getByRole('button', { name: 'More options' }))
    const studioSeed = screen.getByLabelText('LLM random seed')
    expect(studioSeed).toHaveValue(123456)
    await user.clear(studioSeed)
    expect(window.localStorage.getItem('nomas-prompt-enhancer-seed')).toBeNull()

    await user.type(
      screen.getByLabelText('Direct the world'),
      'Rise above the cliffs{Enter}',
    )
    await vi.waitFor(() => {
      expect(
        vi.mocked(fetch).mock.calls.filter(
          ([url]) => String(url).endsWith('v1/generate'),
        ),
      ).toHaveLength(2)
    })
    generateCalls = vi.mocked(fetch).mock.calls.filter(
      ([url]) => String(url).endsWith('v1/generate'),
    )
    body = JSON.parse(
      (generateCalls[1][1] as RequestInit).body as string,
    ) as { pe_input: { prompt_enhancer_seed?: number } }
    expect(body.pe_input).not.toHaveProperty('prompt_enhancer_seed')
  })

  it('migrates the legacy third-person value to third-person follow', () => {
    window.localStorage.setItem('nomas-world-perspective', 'third-person')

    render(<App />)

    expect(normalizePerspective('third-person')).toBe('third-person follow')
    expect(screen.getByLabelText('Perspective')).toHaveValue(
      'third-person follow',
    )
    expect(window.localStorage.getItem('nomas-world-perspective')).toBe(
      'third-person follow',
    )
  })

  it.each([
    ['Third person', 'third-person follow'],
    ['First person', 'first-person'],
    ['Second person (talking head)', 'second-person'],
    ['Third person observer', 'third-person observer'],
  ])('sends the exact %s perspective enum', async (label, perspective) => {
    vi.mocked(fetch).mockImplementation(async (url) => {
      const path = String(url)
      if (path.endsWith('v1/config')) {
        return jsonResponse({
          rollout_type: 'seg_ar',
          prompt_source: 'prompt',
          num_frames: 49,
          num_frames_source: 'request_or_config',
          fps: 24,
          worker_online: true,
          prompt_enhancement_enabled: true,
        })
      }
      if (path.endsWith('v1/generate')) {
        return jsonResponse({
          id: `gen-${perspective}`,
          status: 'succeeded',
          video_url: `outputs/${perspective}.mp4`,
          frames: 120,
          fps: 24,
          generation_seconds: 1,
        })
      }
      return jsonResponse({ status: 'ok' })
    })

    render(<App />)
    const user = userEvent.setup()
    const selector = screen.getByLabelText('Perspective')
    expect(within(selector).getByRole('option', { name: label })).toHaveValue(
      perspective,
    )
    await user.selectOptions(selector, perspective)
    expect(selector).toHaveValue(perspective)
    await user.type(screen.getByLabelText('Subject Description'), 'A red kite')
    await user.type(
      screen.getByLabelText('Environment Description'),
      'A windy coastal meadow',
    )
    await user.click(screen.getByRole('button', { name: 'Enter world' }))

    await vi.waitFor(() => {
      expect(
        vi.mocked(fetch).mock.calls.some(
          ([url]) => String(url).endsWith('v1/generate'),
        ),
      ).toBe(true)
    })
    const call = vi.mocked(fetch).mock.calls.find(
      ([url]) => String(url).endsWith('v1/generate'),
    )
    const body = JSON.parse(
      (call?.[1] as RequestInit).body as string,
    ) as {
      num_inference_steps: number
      pe_input: { perspective: string }
    }
    expect(body.pe_input.perspective).toBe(perspective)
    expect(body.num_inference_steps).toBe(8)
  })

  it('persists inference steps and sends the selected value on every turn', async () => {
    window.localStorage.setItem('nomas-num-inference-steps', '6')
    let generated = 0
    vi.mocked(fetch).mockImplementation(async (url) => {
      const path = String(url)
      if (path.endsWith('v1/config')) {
        return jsonResponse({
          rollout_type: 'seg_ar',
          prompt_source: 'prompt',
          num_frames: 49,
          num_frames_source: 'request_or_config',
          fps: 24,
          worker_online: true,
          prompt_enhancement_enabled: true,
        })
      }
      if (path.endsWith('v1/generate')) {
        generated += 1
        return jsonResponse({
          id: `gen-steps-${generated}`,
          status: 'succeeded',
          video_url: `outputs/steps-${generated}.mp4`,
          frames: generated * 120,
          segment_frames: 120,
          fps: 24,
          generation_seconds: 1,
        })
      }
      return jsonResponse({ status: 'ok' })
    })

    render(<App />)
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'More options' }))
    const setupSteps = screen.getByLabelText('Inference steps')
    expect(setupSteps).toHaveValue('6')
    expect(within(setupSteps).getAllByRole('option')).toHaveLength(3)
    await user.selectOptions(setupSteps, '4')
    expect(window.localStorage.getItem('nomas-num-inference-steps')).toBe('4')
    await user.click(screen.getByRole('button', { name: 'More options' }))

    await user.type(screen.getByLabelText('Subject Description'), 'A red kite')
    await user.type(
      screen.getByLabelText('Environment Description'),
      'A windy coastal meadow',
    )
    await user.click(screen.getByRole('button', { name: 'Enter world' }))
    await screen.findByText('120 frames')

    let generateCalls = vi.mocked(fetch).mock.calls.filter(
      ([url]) => String(url).endsWith('v1/generate'),
    )
    let body = JSON.parse(
      (generateCalls[0][1] as RequestInit).body as string,
    ) as { num_inference_steps: number }
    expect(body.num_inference_steps).toBe(4)

    await user.click(screen.getByRole('button', { name: 'More options' }))
    const studioSteps = screen.getByLabelText('Inference steps')
    expect(studioSteps).toHaveValue('4')
    await user.selectOptions(studioSteps, '8')
    expect(window.localStorage.getItem('nomas-num-inference-steps')).toBe('8')

    await user.type(
      screen.getByLabelText('Direct the world'),
      'Rise above the cliffs{Enter}',
    )
    await vi.waitFor(() => {
      expect(
        vi.mocked(fetch).mock.calls.filter(
          ([url]) => String(url).endsWith('v1/generate'),
        ),
      ).toHaveLength(2)
    })
    generateCalls = vi.mocked(fetch).mock.calls.filter(
      ([url]) => String(url).endsWith('v1/generate'),
    )
    body = JSON.parse(
      (generateCalls[1][1] as RequestInit).body as string,
    ) as { num_inference_steps: number }
    expect(body.num_inference_steps).toBe(8)
  })

  it('uses backend inference step options and default without a stored choice', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({
        rollout_type: 'seg_ar',
        prompt_source: 'prompt',
        num_frames: 49,
        num_frames_source: 'request_or_config',
        fps: 24,
        worker_online: true,
        num_inference_steps: 8,
        num_inference_steps_options: [4, 8],
        default_num_inference_steps: 4,
      }),
    )

    render(<App />)
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'More options' }))

    const selector = screen.getByLabelText('Inference steps')
    expect(selector).toHaveValue('4')
    expect(within(selector).getAllByRole('option')).toHaveLength(2)
    expect(window.localStorage.getItem('nomas-num-inference-steps')).toBe('4')
  })

  it('uses discovered models and per-model thinking options in setup and studio', async () => {
    window.localStorage.setItem(
      'nomas-prompt-enhancer-reasoning-effort',
      'low',
    )
    vi.mocked(fetch)
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            rollout_type: 'seg_ar',
            prompt_source: 'prompt',
            num_frames: 49,
            num_frames_source: 'request_or_config',
            fps: 24,
            worker_online: true,
            prompt_enhancement_enabled: true,
            default_prompt_enhancer_model: 'kimi-k3',
            prompt_enhancer_options: [
              {
                id: 'kimi-k3',
                label: 'Kimi-K3',
                default: true,
                reasoning_effort_options: [
                  { id: 'low', label: 'Low' },
                  { id: 'high', label: 'High', default: true },
                ],
                default_reasoning_effort: 'high',
              },
              {
                id: 'org/Model-A',
                label: 'org/Model-A',
                reasoning_effort_options: [
                  { id: 'none', label: 'Off' },
                  { id: 'medium', label: 'Medium', default: true },
                ],
                default_reasoning_effort: 'medium',
              },
            ],
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )
      .mockResolvedValue(
        new Response(
          JSON.stringify({
            id: 'gen_thinking',
            status: 'succeeded',
            video_url: 'outputs/thinking.mp4',
            frames: 120,
            fps: 24,
            generation_seconds: 1,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )

    render(<App />)
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'More options' }))

    const setupThinking = screen.getByLabelText('Thinking mode')
    expect(setupThinking).toHaveValue('low')
    expect(within(setupThinking).getAllByRole('option')).toHaveLength(2)

    await user.selectOptions(setupThinking, 'high')
    expect(setupThinking).toHaveValue('high')
    expect(window.localStorage.getItem(
      'nomas-prompt-enhancer-reasoning-effort',
    )).toBe('high')

    const setupModel = screen.getByLabelText('Prompt enhancer model')
    expect(within(setupModel).getAllByRole('option')).toHaveLength(2)
    await user.selectOptions(setupModel, 'org/Model-A')
    expect(setupThinking).toHaveValue('medium')
    expect(window.localStorage.getItem('nomas-prompt-enhancer-model')).toBe('org/Model-A')

    await user.click(screen.getByRole('button', { name: 'More options' }))
    await user.type(
      screen.getByLabelText('Subject Description'),
      'A brass firefly',
    )
    await user.type(
      screen.getByLabelText('Environment Description'),
      'A moonlit glasshouse',
    )
    await user.click(screen.getByRole('button', { name: 'Enter world' }))

    await vi.waitFor(() => {
      expect(
        vi.mocked(fetch).mock.calls.some(
          ([url]) => url === `${import.meta.env.BASE_URL}v1/generate`,
        ),
      ).toBe(true)
    })
    const initialGenerate = vi.mocked(fetch).mock.calls.find(
      ([url]) => url === `${import.meta.env.BASE_URL}v1/generate`,
    )
    const initialBody = JSON.parse(
      (initialGenerate?.[1] as RequestInit).body as string,
    ) as {
      pe_input: {
        prompt_enhancer_model: string
        prompt_enhancer_reasoning_effort: string
      }
    }
    expect(initialBody.pe_input).toMatchObject({
      prompt_enhancer_model: 'org/Model-A',
      prompt_enhancer_reasoning_effort: 'medium',
    })

    await user.click(screen.getByRole('button', { name: 'More options' }))
    const studioThinking = screen.getByLabelText('Thinking mode')
    expect(screen.getByLabelText('Prompt enhancer model')).toHaveValue('org/Model-A')
    await user.selectOptions(screen.getByLabelText('Prompt enhancer model'), 'kimi-k3')
    expect(studioThinking).toHaveValue('high')
    await user.selectOptions(studioThinking, 'low')
    expect(window.localStorage.getItem(
      'nomas-prompt-enhancer-reasoning-effort',
    )).toBe('low')

    await user.type(
      screen.getByLabelText('Direct the world'),
      'Fly toward the moon{Enter}',
    )
    await vi.waitFor(() => {
      expect(
        vi.mocked(fetch).mock.calls.filter(
          ([url]) => url === `${import.meta.env.BASE_URL}v1/generate`,
        ),
      ).toHaveLength(2)
    })
    const generateCalls = vi.mocked(fetch).mock.calls.filter(
      ([url]) => url === `${import.meta.env.BASE_URL}v1/generate`,
    )
    const followUpBody = JSON.parse(
      (generateCalls[1][1] as RequestInit).body as string,
    ) as {
      pe_input: {
        prompt_enhancer_model: string
        prompt_enhancer_reasoning_effort: string
      }
    }
    expect(followUpBody.pe_input.prompt_enhancer_model).toBe('kimi-k3')
    expect(followUpBody.pe_input.prompt_enhancer_reasoning_effort).toBe('low')
  })

  it('releases the actual inference session when the world page closes', async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            rollout_type: 'seg_ar',
            prompt_source: 'prompt',
            num_frames: 49,
            num_frames_source: 'request_or_config',
            fps: 24,
            worker_online: true,
            prompt_enhancement_enabled: false,
            prompt_fallback_enabled: true,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'gen_release',
            status: 'succeeded',
            video_url: 'outputs/release.mp4',
            frames: 120,
            fps: 24,
            generation_seconds: 1,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )
      .mockResolvedValue(
        new Response(JSON.stringify({ status: 'succeeded' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )

    const rendered = render(<App />)
    await enterWorld(
      'A survey robot inside a lunar tunnel',
      'The robot switches on its headlamp',
    )

    await vi.waitFor(() => {
      expect(
        vi.mocked(fetch).mock.calls.some(([url]) =>
          String(url).endsWith('v1/generate'),
        ),
      ).toBe(true)
    })
    const generateCall = vi.mocked(fetch).mock.calls.find(([url]) =>
      String(url).endsWith('v1/generate'),
    )
    const request = JSON.parse(
      (generateCall?.[1] as RequestInit).body as string,
    ) as {
      session_id: string
      pe_input: { session_id: string }
    }

    rendered.unmount()

    expect(request.session_id).toBeTruthy()
    expect(request.session_id).toBe(request.pe_input.session_id)
    expect(vi.mocked(fetch)).toHaveBeenCalledWith(
      `${import.meta.env.BASE_URL}v1/sessions/${encodeURIComponent(request.session_id)}/release`,
      expect.objectContaining({
        method: 'POST',
        keepalive: true,
        headers: {
          'X-Nomas-Explicit-Session-Release': '1',
        },
      }),
    )
  })

  it('preserves the inference session when pagehide is followed by unmount', async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            rollout_type: 'seg_ar',
            prompt_source: 'prompt',
            num_frames: 49,
            num_frames_source: 'request_or_config',
            fps: 24,
            worker_online: true,
            prompt_enhancement_enabled: false,
            prompt_fallback_enabled: true,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'gen_refresh',
            status: 'succeeded',
            video_url: 'outputs/refresh.mp4',
            frames: 120,
            fps: 24,
            generation_seconds: 1,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )
      .mockResolvedValue(
        new Response(JSON.stringify({ status: 'succeeded' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )

    const rendered = render(<App />)
    await enterWorld(
      'A survey robot inside a lunar tunnel',
      'The browser refreshes while the world remains resident',
    )

    await vi.waitFor(() => {
      expect(
        vi.mocked(fetch).mock.calls.some(([url]) =>
          String(url).endsWith('v1/generate'),
        ),
      ).toBe(true)
    })

    fireEvent(window, new PageTransitionEvent('pagehide'))
    rendered.unmount()

    expect(
      vi.mocked(fetch).mock.calls.some(([url]) =>
        /\/v1\/sessions\/[^/]+\/release$/.test(String(url)),
      ),
    ).toBe(false)
  })

  it('starts segment 1 immediately from the environment description', async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            rollout_type: 'seg_ar',
            prompt_source: 'prompt',
            num_frames: 49,
            num_frames_source: 'request_or_config',
            fps: 24,
            worker_online: true,
            prompt_enhancement_enabled: false,
            prompt_fallback_enabled: true,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'gen_subject',
            status: 'succeeded',
            video_url: 'outputs/subject.mp4',
            frames: 120,
            fps: 24,
            generation_seconds: 1,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )

    render(<App />)
    const user = userEvent.setup()
    await user.type(screen.getByLabelText('Subject Description'), 'A silver fox')
    await user.type(screen.getByLabelText('Environment Description'), 'Walk into the greenhouse')
    const enterButton = await screen.findByRole('button', { name: 'Enter world' })
    expect(enterButton).toBeEnabled()
    await user.click(enterButton)

    expect(screen.getByLabelText('World model output')).toBeInTheDocument()
    expect(screen.getAllByText('A silver fox')).not.toHaveLength(0)
    expect(await screen.findByText('120 frames')).toBeInTheDocument()
    const generateCall = vi.mocked(fetch).mock.calls.find(
      ([url]) => url === `${import.meta.env.BASE_URL}v1/generate`,
    )
    expect(generateCall).toBeDefined()
    const body = JSON.parse(
      (generateCall?.[1] as RequestInit).body as string,
    ) as {
      pe_input: {
        perspective: string
        subject_description: string
        environment_description: string
      }
    }
    expect(body.pe_input).toMatchObject({
      perspective: 'first-person',
      subject_description: 'A silver fox',
      environment_description: 'Walk into the greenhouse',
    })
    expect(screen.queryByRole('button', { name: /audio/i })).not.toBeInTheDocument()
  })

  it('keeps rollback disabled until a generated segment exists', async () => {
    render(<App />)
    const user = await enterWorld()

    expect(
      screen.queryByRole('button', { name: 'Redo last' }),
    ).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'More options' }))
    expect(
      screen.getByRole('menuitem', { name: /Redo last segment/ }),
    ).toBeDisabled()
  })

  it('rolls back the latest segment, restores its prompt, and regenerates it', async () => {
    const generation = (
      id: string,
      videoUrl: string,
      frames: number,
      playbackStartSeconds: number,
    ) => new Response(
      JSON.stringify({
        id,
        status: 'succeeded',
        video_url: videoUrl,
        frames,
        segment_frames: 120,
        fps: 24,
        generation_seconds: 1,
        playback_start_seconds: playbackStartSeconds,
        prompt_enhanced: true,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      },
    )
    let generationCount = 0
    vi.mocked(fetch).mockImplementation(async (url, options) => {
      const path = String(url)
      if (path.endsWith('v1/config')) {
        return new Response(
          JSON.stringify({
            rollout_type: 'seg_ar',
            prompt_source: 'prompt',
            num_frames: 49,
            num_frames_source: 'request_or_config',
            fps: 24,
            worker_online: true,
            prompt_enhancement_enabled: true,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        )
      }
      if (path.includes('/rollback/')) {
        return new Response(
          JSON.stringify({
            id: 'rollback_latest',
            status: 'succeeded',
            result: {
              session_id: 'session-test',
              remaining_segments: 1,
              removed_cache_tokens: 100,
              removed_latent_frames: 30,
              video_url: 'outputs/initial.mp4',
              frames: 120,
              fps: 24,
            },
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        )
      }
      if (path.endsWith('/rollback')) {
        return new Response(
          JSON.stringify({
            id: 'rollback_latest',
            status: 'queued',
          }),
          {
            status: 202,
            headers: { 'Content-Type': 'application/json' },
          },
        )
      }
      if (
        path.endsWith('v1/generate')
        && (options as RequestInit | undefined)?.method === 'POST'
      ) {
        generationCount += 1
        return generationCount === 1
          ? generation('gen_initial', 'outputs/initial.mp4', 120, 0)
          : generationCount === 2
            ? generation('gen_bad', 'outputs/bad.mp4', 240, 5)
            : generation(
                'gen_replacement',
                'outputs/replacement.mp4',
                240,
                5,
              )
      }
      return new Response(JSON.stringify({ status: 'ok' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    })

    render(<App />)
    const user = await enterWorld(
      'An astronaut in a white pressure suit',
      'A dim spacecraft corridor',
    )
    await user.click(screen.getByRole('button', { name: 'More options' }))
    await vi.waitFor(() => {
      expect(
        screen.getByRole('menuitem', { name: /Redo last segment/ }),
      ).toBeEnabled()
    })
    await user.keyboard('{Escape}')

    const composer = screen.getByLabelText('Direct the world')
    await user.type(composer, 'Turn left into the dark room{Enter}')
    await user.click(screen.getByRole('button', { name: 'More options' }))
    await vi.waitFor(() => {
      expect(
        screen.getByRole('menuitem', { name: /Redo last segment/ }),
      ).toBeEnabled()
    })
    await user.click(
      screen.getByRole('menuitem', { name: /Redo last segment/ }),
    )

    await vi.waitFor(() => {
      expect(screen.getByLabelText('Direct the world')).toHaveValue(
        'Turn left into the dark room',
      )
    })
    expect(screen.queryByText('PROMPT 2')).not.toBeInTheDocument()

    await user.clear(screen.getByLabelText('Direct the world'))
    await user.type(
      screen.getByLabelText('Direct the world'),
      'Turn right toward the window{Enter}',
    )
    await vi.waitFor(() => {
      expect(generationCount).toBe(3)
    })

    const generationCalls = vi.mocked(fetch).mock.calls.filter(
      ([url, options]) =>
        url === `${import.meta.env.BASE_URL}v1/generate`
        && (options as RequestInit | undefined)?.method === 'POST',
    )
    expect(generationCalls).toHaveLength(3)
    const replacement = JSON.parse(
      (generationCalls[2][1] as RequestInit).body as string,
    ) as {
      reset_session: boolean
      pe_input: { action: string }
    }
    expect(replacement.reset_session).toBe(false)
    expect(replacement.pe_input.action).toBe('Turn right toward the window')
  })

  it('keeps reconnecting after repeated generation polling failures', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    let pollCount = 0
    vi.mocked(fetch).mockImplementation(async (url, options) => {
      const path = String(url)
      if (path.endsWith('v1/config')) {
        return new Response(
          JSON.stringify({
            rollout_type: 'seg_ar',
            prompt_source: 'prompt',
            num_frames: 49,
            num_frames_source: 'request_or_config',
            fps: 24,
            worker_online: true,
            prompt_enhancement_enabled: true,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        )
      }
      if (
        path.endsWith('v1/generate')
        && (options as RequestInit | undefined)?.method === 'POST'
      ) {
        return new Response(
          JSON.stringify({ id: 'gen_retry', status: 'enhancing' }),
          {
            status: 202,
            headers: { 'Content-Type': 'application/json' },
          },
        )
      }
      if (path.endsWith('v1/generate/gen_retry')) {
        pollCount += 1
        if (pollCount <= 9) {
          throw new TypeError('Failed to fetch')
        }
        return new Response(
          JSON.stringify({
            id: 'gen_retry',
            status: 'succeeded',
            result: {
              id: 'gen_retry_result',
              status: 'succeeded',
              video_url: 'outputs/retry.mp4',
              frames: 120,
              segment_frames: 120,
              fps: 24,
              generation_seconds: 1,
              playback_start_seconds: 0,
              prompt_enhanced: true,
            },
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        )
      }
      return new Response(JSON.stringify({ status: 'ok' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    })

    render(<App />)
    const user = userEvent.setup({
      advanceTimers: vi.advanceTimersByTime,
    })
    await user.type(
      screen.getByLabelText('Subject Description'),
      'A silver fox',
    )
    await user.type(
      screen.getByLabelText('Environment Description'),
      'A floating greenhouse',
    )
    await user.click(screen.getByRole('button', { name: 'Enter world' }))
    await vi.advanceTimersByTimeAsync(40_000)

    await vi.waitFor(() => {
      expect(pollCount).toBe(10)
      expect(screen.getByLabelText('Direct the world')).toBeEnabled()
    })
    expect(screen.queryByText('Failed to fetch')).not.toBeInTheDocument()
    vi.useRealTimers()
  }, 20_000)

  it('keeps reconnecting when a generation job is temporarily not found', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    let pollCount = 0
    vi.mocked(fetch).mockImplementation(async (url, options) => {
      const path = String(url)
      if (path.endsWith('v1/config')) {
        return new Response(
          JSON.stringify({
            rollout_type: 'seg_ar',
            prompt_source: 'prompt',
            num_frames: 49,
            num_frames_source: 'request_or_config',
            fps: 24,
            worker_online: true,
            prompt_enhancement_enabled: true,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        )
      }
      if (
        path.endsWith('v1/generate')
        && (options as RequestInit | undefined)?.method === 'POST'
      ) {
        return new Response(
          JSON.stringify({ id: 'gen_cross_host', status: 'enhancing' }),
          {
            status: 202,
            headers: { 'Content-Type': 'application/json' },
          },
        )
      }
      if (path.endsWith('v1/generate/gen_cross_host')) {
        pollCount += 1
        if (pollCount === 1) {
          return new Response(
            JSON.stringify({ detail: 'generation job not found' }),
            {
              status: 404,
              headers: { 'Content-Type': 'application/json' },
            },
          )
        }
        return new Response(
          JSON.stringify({
            id: 'gen_cross_host',
            status: 'succeeded',
            result: {
              id: 'gen_cross_host_result',
              status: 'succeeded',
              video_url: 'outputs/cross-host.mp4',
              frames: 120,
              segment_frames: 120,
              fps: 24,
              generation_seconds: 1,
              playback_start_seconds: 0,
              prompt_enhanced: true,
            },
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        )
      }
      return new Response(JSON.stringify({ status: 'ok' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    })

    render(<App />)
    const user = userEvent.setup({
      advanceTimers: vi.advanceTimersByTime,
    })
    await user.type(
      screen.getByLabelText('Subject Description'),
      'A silver fox',
    )
    await user.type(
      screen.getByLabelText('Environment Description'),
      'A floating greenhouse',
    )
    await user.click(screen.getByRole('button', { name: 'Enter world' }))
    await vi.advanceTimersByTimeAsync(5_000)

    await vi.waitFor(() => {
      expect(pollCount).toBe(2)
      expect(screen.getByLabelText('Direct the world')).toBeEnabled()
    })
    expect(
      screen.queryByText(/generation job not found/i),
    ).not.toBeInTheDocument()
    vi.useRealTimers()
  }, 20_000)

  it('automatically submits the initial environment after entering the world', async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            rollout_type: 'seg_ar',
            prompt_source: 'prompt',
            num_frames: 49,
            num_frames_source: 'request_or_config',
            fps: 24,
            worker_online: true,
            prompt_enhancement_enabled: true,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'gen_initial',
            status: 'succeeded',
            video_url: 'outputs/initial.mp4',
            frames: 120,
            fps: 24,
            generation_seconds: 1,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )

    render(<App />)
    const user = userEvent.setup()
    await user.type(screen.getByLabelText('Subject Description'), 'A silver fox')
    await user.type(screen.getByLabelText('Environment Description'), 'Walk toward the river')
    await user.click(screen.getByRole('button', { name: 'Enter world' }))

    expect(await screen.findAllByText('Walk toward the river')).not.toHaveLength(0)
    const generateCall = vi.mocked(fetch).mock.calls.find(
      ([url]) => url === `${import.meta.env.BASE_URL}v1/generate`,
    )
    expect(generateCall).toBeDefined()
    const body = JSON.parse(
      (generateCall?.[1] as RequestInit).body as string,
    ) as { pe_input: { environment_description: string } }
    expect(body.pe_input.environment_description).toBe('Walk toward the river')
    expect(
      screen.queryByText(
        'Initial direction added to the prompt sequence. The renderer connection will execute it when the backend is available.',
      ),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByText('Change to golden-hour light'),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByText('The subject looks to the left'),
    ).not.toBeInTheDocument()
  })

  it('creates a public video link from the generated result', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    })
    vi.mocked(fetch)
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            rollout_type: 'seg_ar',
            prompt_source: 'prompt',
            num_frames: 49,
            num_frames_source: 'request_or_config',
            fps: 24,
            worker_online: true,
            prompt_enhancement_enabled: false,
            prompt_fallback_enabled: true,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'gen_share',
            status: 'succeeded',
            video_url: 'outputs/share-me.mp4',
            frames: 120,
            fps: 24,
            generation_seconds: 1,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'public-token',
            share_path: './share/public-token',
          }),
          {
            status: 201,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )

    render(<App />)
    const user = await enterWorld(
      'A silver fox',
      'Walk through the greenhouse',
    )
    const share = await screen.findByRole('button', { name: 'Share' })
    expect(share).toBeEnabled()
    await user.click(share)

    expect(await screen.findByRole('dialog', {
      name: 'Ready to share',
    })).toBeInTheDocument()
    expect(screen.getByLabelText('Public share link')).toHaveValue(
      'http://localhost:3000/share/public-token',
    )
    const shareCall = vi.mocked(fetch).mock.calls.find(
      ([url]) => url === `${import.meta.env.BASE_URL}v1/shares`,
    )
    expect(shareCall?.[1]).toEqual(expect.objectContaining({ method: 'POST' }))
    const shareBody = JSON.parse(
      (shareCall?.[1] as RequestInit).body as string,
    ) as {
      video_url: string
      overlay: {
        version: number
        turns: Array<{
          prompt: string
          start_seconds: number
          end_seconds: number
        }>
      }
    }
    expect(shareBody.video_url).toBe(
      'outputs/share-me.mp4?request=gen_share',
    )
    expect(shareBody.overlay).toMatchObject({
      version: 1,
      turns: [{
        prompt: 'Walk through the greenhouse',
        start_seconds: 0,
        end_seconds: 5,
      }],
    })
  })

  it('defaults downloads to a video with prompt and camera overlays', async () => {
    const anchorClick = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => undefined)
    vi.mocked(fetch)
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            rollout_type: 'seg_ar',
            prompt_source: 'prompt',
            num_frames: 49,
            num_frames_source: 'request_or_config',
            fps: 24,
            worker_online: true,
            prompt_enhancement_enabled: false,
            prompt_fallback_enabled: true,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'gen_download',
            status: 'succeeded',
            video_url: 'outputs/download-me.mp4',
            frames: 120,
            fps: 24,
            generation_seconds: 1,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            download_path:
              './outputs/downloads/download-me-with-overlay-test.mp4',
            filename: 'download-me-with-overlay-test.mp4',
            overlay: true,
          }),
          {
            status: 201,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )

    render(<App />)
    const user = await enterWorld(
      'A silver fox',
      'Walk through the greenhouse',
    )
    const download = await screen.findByRole('button', { name: 'Download' })
    await user.click(download)

    expect(screen.getByRole('dialog', {
      name: 'Choose a version',
    })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /With overlay/ })).toBeChecked()
    expect(screen.getByRole('radio', { name: /Original video/ })).not.toBeChecked()

    await user.click(
      screen.getByRole('button', { name: 'Download with overlay' }),
    )

    const downloadCall = vi.mocked(fetch).mock.calls.find(
      ([url]) => url === `${import.meta.env.BASE_URL}v1/downloads`,
    )
    expect(downloadCall?.[1]).toEqual(expect.objectContaining({ method: 'POST' }))
    const downloadBody = JSON.parse(
      (downloadCall?.[1] as RequestInit).body as string,
    ) as {
      video_url: string
      include_overlay: boolean
      overlay: { version: number }
    }
    expect(downloadBody.video_url).toBe(
      'outputs/download-me.mp4?request=gen_download',
    )
    expect(downloadBody.include_overlay).toBe(true)
    expect(downloadBody.overlay.version).toBe(1)
    expect(anchorClick).toHaveBeenCalledOnce()
    anchorClick.mockRestore()
  })

  it('shows the camera recorder without leaving the studio', async () => {
    render(<App />)
    const user = await enterWorld()

    await user.click(screen.getByRole('button', { name: 'Camera path' }))
    expect(screen.getByText('Record in 3D')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Import JSON' })).toBeInTheDocument()
    expect(screen.queryByText(/LIMIT/)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Generate with camera path' })).toBeDisabled()
    expect(screen.queryByLabelText('Direct the world')).not.toBeInTheDocument()
  })

  it.each([false, true])('uses 24 FPS camera timing and request-level scales (override=%s)', async (overrideScales) => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            rollout_type: 'seg_ar',
            prompt_source: 'prompt',
            num_frames: 49,
            num_frames_source: 'request_or_config',
            fps: 24,
            trajectory_translation_scale: 1.3,
            trajectory_rotation_scale: 2.1,
            prompt_enhancement_enabled: true,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'gen_camera',
            status: 'succeeded',
            video_url: 'outputs/camera.mp4',
            frames: 66,
            segment_frames: 66,
            fps: 24,
            generation_seconds: 1,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )

    render(<App />)
    const user = await enterWorld()
    await user.click(screen.getByRole('button', { name: 'Camera path' }))

    const frames = Array.from({ length: 66 }, (_, index) => ({
      frame: index,
      t: Math.round(index * 1000 / 24),
      position: { x: index / 100, y: 0, z: 0 },
      quaternion: { x: 0, y: 0, z: 0, w: 1 },
      fov: 55,
    }))
    await user.upload(
      screen.getByLabelText('Camera trajectory JSON'),
      new File(
        [JSON.stringify({
          name: '24 FPS path', sampleRate: 24, frames,
          trajectory_translation_scale: 9, trajectory_rotation_scale: 8,
          translationScale: 7, rotationScale: 6,
          keyEvents: [
            { t: 0, code: 'KeyW', pressed: true },
            { t: 500, code: 'ArrowLeft', pressed: true },
            { t: 2763, code: 'KeyW', pressed: false },
            { t: 2800, code: 'KeyD', pressed: true },
          ],
        })],
        'camera.json',
        { type: 'application/json' },
      ),
    )
    expect(screen.getByLabelText('Camera translation scale')).toHaveValue(1.3)
    expect(screen.getByLabelText('Camera rotation scale')).toHaveValue(2.1)
    if (overrideScales) {
      await user.clear(screen.getByLabelText('Camera translation scale'))
      await user.type(screen.getByLabelText('Camera translation scale'), '1.7')
      await user.clear(screen.getByLabelText('Camera rotation scale'))
      await user.type(screen.getByLabelText('Camera rotation scale'), '3.2')
    }
    await user.click(screen.getByRole('button', { name: 'Generate with camera path' }))

    const generateCall = vi.mocked(fetch).mock.calls.find(
      ([url]) => url === `${import.meta.env.BASE_URL}v1/generate`,
    )
    expect(generateCall).toBeDefined()
    const body = JSON.parse(
      (generateCall?.[1] as RequestInit).body as string,
    ) as {
      pe_input: {
        camera_duration_ms: number
        camera_controlled: boolean
        action?: string
        subject_description: string
        environment_description: string
        perspective: string
      }
      segment_has_camera: boolean[]
      camera_trajectory: string
      trajectory_translation_scale: number
      trajectory_rotation_scale: number
      num_inference_steps: number
    }
    expect(body.pe_input.camera_duration_ms).toBeCloseTo(2750)
    expect(body.pe_input.camera_controlled).toBe(true)
    expect(body.pe_input.action).toBeUndefined()
    expect(body.pe_input.subject_description).toBeTruthy()
    expect(body.pe_input.environment_description).toBeTruthy()
    expect(body.pe_input.perspective).toBe('first-person')
    expect(body.pe_input).not.toHaveProperty('style')
    expect(body.segment_has_camera).toEqual([true])
    expect(body.num_inference_steps).toBe(8)
    expect(body.trajectory_translation_scale).toBe(overrideScales ? 1.7 : 1.3)
    expect(body.trajectory_rotation_scale).toBe(overrideScales ? 3.2 : 2.1)
    expect(JSON.parse(body.camera_trajectory)).toMatchObject({
      sampleRate: 24,
      durationMs: 2750,
      trajectory_translation_scale: 9,
      trajectory_rotation_scale: 8,
      frames,
      keyEvents: [
        { t: 0, code: 'KeyW', pressed: true },
        { t: 500, code: 'ArrowLeft', pressed: true },
        { t: 2750, code: 'KeyW', pressed: false },
        { t: 2750, code: 'ArrowLeft', pressed: false },
      ],
    })
    expect(JSON.parse(body.camera_trajectory)).not.toHaveProperty('translationScale')
    expect(JSON.parse(body.camera_trajectory)).not.toHaveProperty('rotationScale')
    expect(screen.getByRole('button', { name: 'Generate with camera path' })).toBeDisabled()
    expect(screen.queryByText(/Ready to generate/)).not.toBeInTheDocument()
  })

  it('returns to setup without losing the configuration', async () => {
    render(<App />)
    const user = await enterWorld('A robot gardener')

    await user.click(screen.getByRole('button', { name: 'More options' }))
    expect(
      screen.queryByRole('menuitem', { name: /New world/ }),
    ).not.toBeInTheDocument()
    await user.keyboard('{Escape}')

    await user.click(screen.getByRole('button', { name: 'New world' }))

    expect(screen.getByLabelText('Subject Description')).toHaveValue('A robot gardener')
    expect(
      vi.mocked(fetch).mock.calls.some(([url]) =>
        String(url).endsWith('auth/logout'),
      ),
    ).toBe(false)
  })

  it('labels playback restart as replay from start', async () => {
    render(<App />)
    await enterWorld()

    expect(
      screen.getByRole('button', { name: 'Replay from start' }),
    ).toBeDisabled()
    expect(screen.queryByRole('button', { name: 'Restart' }))
      .not.toBeInTheDocument()
  })

  it('shows a prompt-file input when the backend runs in oneshot mode', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          rollout_type: 'oneshot',
          prompt_source: 'subject_file',
          num_frames: null,
          num_frames_source: 'prompt_timestamps',
          fps: 24,
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        },
      ),
    )

    render(<App />)

    expect(await screen.findByLabelText('Prompt file')).toHaveAttribute(
      'placeholder',
      '/path/to/prompt.txt',
    )
  })

  it('computes seg_ar boundaries from structured JSON timestamps', async () => {
    const generateResponse = new Response(
      JSON.stringify({
        id: 'gen_test',
        status: 'succeeded',
        video_url: 'outputs/test.mp4',
        frames: 84,
        fps: 24,
        generation_seconds: 1,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      },
    )
    vi.mocked(fetch)
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            rollout_type: 'seg_ar',
            prompt_source: 'prompt',
            num_frames: 49,
            num_frames_source: 'request_or_config',
            fps: 24,
            worker_online: true,
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )
      .mockResolvedValueOnce(generateResponse)

    render(<App />)
    const user = userEvent.setup()
    const structuredPrompt = JSON.stringify({
      perspective_view: 'first-person',
      segments: [
        {
          visual_description_long: 'A fox stands still.',
          contained_actions: [
            { start_time: '00:00.000', end_time: '00:02.000' },
          ],
        },
        {
          description_change: 'The fox walks forward.',
          contained_actions: [
            { start_time: '00:00.000', end_time: '00:01.500' },
          ],
        },
      ],
    })
    fireEvent.change(screen.getByLabelText('Subject Description'), {
      target: { value: structuredPrompt },
    })
    await user.type(screen.getByLabelText('Environment Description'), 'Generate')
    await user.click(screen.getByRole('button', { name: 'Enter world' }))

    expect(vi.mocked(fetch)).toHaveBeenLastCalledWith(
      `${import.meta.env.BASE_URL}v1/generate`,
      expect.objectContaining({
        method: 'POST',
        body: expect.any(String),
      }),
    )
    const lastCall = vi.mocked(fetch).mock.calls.at(-1)
    const body = JSON.parse(
      (lastCall?.[1] as RequestInit).body as string,
    ) as {
      prompt: string
      num_frames: number
      segment_boundaries: number[]
      num_inference_steps: number
    }
    expect(body.num_frames).toBe(84)
    expect(body.segment_boundaries).toEqual([0, 48, 84])
    expect(body.num_inference_steps).toBe(8)
    expect(body.prompt.split('|||')).toHaveLength(2)
  })
})

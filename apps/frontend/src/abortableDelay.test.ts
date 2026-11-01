import { afterEach, expect, it, vi } from 'vitest'
import { abortableDelay } from './abortableDelay'

afterEach(() => vi.useRealTimers())

it('clears a pending polling timer when the world exits', async () => {
  vi.useFakeTimers()
  const controller = new AbortController()
  const pending = abortableDelay(1500, controller.signal)
  const cancelled = expect(pending).rejects.toMatchObject({ name: 'AbortError' })
  expect(vi.getTimerCount()).toBe(1)
  controller.abort()
  await cancelled
  expect(vi.getTimerCount()).toBe(0)
})

it('completes an active world polling delay', async () => {
  vi.useFakeTimers()
  const controller = new AbortController()
  const pending = abortableDelay(500, controller.signal)
  await vi.advanceTimersByTimeAsync(500)
  await expect(pending).resolves.toBeUndefined()
  expect(vi.getTimerCount()).toBe(0)
})

it('does not schedule polling for a world that already exited', () => {
  vi.useFakeTimers()
  const controller = new AbortController()
  controller.abort()
  expect(() => abortableDelay(500, controller.signal)).toThrow()
  expect(vi.getTimerCount()).toBe(0)
})

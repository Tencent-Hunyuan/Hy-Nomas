import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'
import { DemoOverlayTimeline } from './DemoOverlayTimeline'
import type { PromptTurn } from './PromptTurnTimeline'

afterEach(cleanup)

it('labels the half-second reference T0 and the following text prompt T1', () => {
  render(<DemoOverlayTimeline currentTime={0.6} turns={[
    { id: 1, number: 0, referenceImage: true, prompt: 'Reference Image', submittedAt: '', stage: 'complete', startSeconds: 0, endSeconds: 0.5 },
    { id: 2, number: 1, prompt: 'The ball rolls.', submittedAt: '', stage: 'playing', startSeconds: 0.5, endSeconds: 2.5 },
  ]} />)
  expect(screen.getByText('T0-Reference Image')).toBeInTheDocument()
  expect(screen.getByText('T1 · TEXT')).toBeInTheDocument()
  expect(screen.queryByText('T1 · START')).not.toBeInTheDocument()
})

it('sizes segments by duration and tracks seeking without a top-left counter', () => {
  const turns: PromptTurn[] = [
    { id: 1, number: 1, prompt: 'Start', submittedAt: '', stage: 'complete', startSeconds: 0, endSeconds: 2 },
    { id: 2, number: 2, prompt: 'Continue', submittedAt: '', stage: 'complete', startSeconds: 2, endSeconds: 6 },
    { id: 3, number: 3, prompt: 'Pending', submittedAt: '', stage: 'rendering' },
  ]
  const { container, rerender } = render(<DemoOverlayTimeline turns={turns} currentTime={4} />)
  expect(screen.getByText('T1 · START')).toBeInTheDocument()
  expect(screen.getByText('T2 · TEXT')).toBeInTheDocument()
  expect(container.querySelectorAll('.demo-timeline-segment')).toHaveLength(2)
  expect(container.querySelectorAll('i')[1]).toHaveStyle({ width: '50%' })
  expect(container.querySelectorAll('b')).toHaveLength(1)
  rerender(<DemoOverlayTimeline turns={turns} currentTime={1} />)
  expect(container.querySelectorAll('i')[0]).toHaveStyle({ width: '50%' })
  expect(container.querySelectorAll('i')[1]).toHaveStyle({ width: '0%' })
})

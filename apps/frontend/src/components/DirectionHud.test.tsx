import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { DirectionHud } from './DirectionHud'

afterEach(cleanup)

describe('DirectionHud', () => {
  it('renders a bottom caption without a title or turn-count card', () => {
    const onVisibilityToggle = vi.fn()
    const { container } = render(
      <DirectionHud
        direction="Follow the river toward the bridge"
        visible
        onVisibilityToggle={onVisibilityToggle}
      />,
    )

    expect(screen.getByLabelText('Real user prompt')).toHaveTextContent(
      'Follow the river toward the bridge',
    )
    expect(screen.getByText('TEXT')).toBeInTheDocument()
    const visibilityToggle = screen.getByRole('button', {
      name: 'Hide real user prompt',
    })
    expect(visibilityToggle).toHaveTextContent('Prompt overlay on')
    expect(visibilityToggle).toHaveAttribute('aria-pressed', 'true')
    expect(screen.queryByText('Turn 02')).not.toBeInTheDocument()
    expect(screen.queryByText('LIVE')).not.toBeInTheDocument()
    expect(container.querySelector('.demo-prompt-overlay')).toBeInTheDocument()
    expect(container.querySelector('.direction-hud-frame')).toBeNull()
  })

  it('offers compact controls to hide and restore the prompt', async () => {
    const user = userEvent.setup()
    const onVisibilityToggle = vi.fn()
    const { rerender } = render(
      <DirectionHud
        direction="Follow the river toward the bridge"
        visible
        onVisibilityToggle={onVisibilityToggle}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'Hide real user prompt' }))
    expect(onVisibilityToggle).toHaveBeenCalledOnce()

    rerender(
      <DirectionHud
        direction="Follow the river toward the bridge"
        visible={false}
        onVisibilityToggle={onVisibilityToggle}
      />,
    )
    expect(screen.queryByLabelText('Real user prompt')).not.toBeInTheDocument()
    const hiddenVisibilityToggle = screen.getByRole('button', {
      name: 'Show real user prompt',
    })
    expect(hiddenVisibilityToggle).toHaveTextContent('Prompt overlay off')
    expect(hiddenVisibilityToggle).toHaveAttribute('aria-pressed', 'false')
    await user.click(screen.getByRole('button', { name: 'Show real user prompt' }))
    expect(onVisibilityToggle).toHaveBeenCalledTimes(2)
  })
})

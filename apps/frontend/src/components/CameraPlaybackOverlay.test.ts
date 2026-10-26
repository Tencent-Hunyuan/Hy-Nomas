import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createElement, createRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  CameraPlaybackOverlay,
  actionsAtTime,
  cameraSegmentIsActive,
} from './CameraPlaybackOverlay'
import type { CameraPath } from '../types'

afterEach(cleanup)

describe('camera playback action display', () => {
  it('replays exact keydown and keyup timestamps without frame quantization', () => {
    const events = [
      { t: 10.25, code: 'KeyW' as const, pressed: true },
      { t: 17.75, code: 'KeyD' as const, pressed: true },
      { t: 23.5, code: 'KeyW' as const, pressed: false },
      { t: 29.125, code: 'ArrowRight' as const, pressed: true },
    ]

    expect(actionsAtTime(events, 10.249).forward).toBe(false)
    expect(actionsAtTime(events, 10.25).forward).toBe(true)
    expect(actionsAtTime(events, 20).right).toBe(true)
    expect(actionsAtTime(events, 23.5).forward).toBe(false)
    expect(actionsAtTime(events, 29.125).lookRight).toBe(true)
  })

  it('shows no actions when a trajectory has no original key events', () => {
    expect(actionsAtTime([], 1000)).toEqual({
      forward: false,
      backward: false,
      left: false,
      right: false,
      lookUp: false,
      lookDown: false,
      lookLeft: false,
      lookRight: false,
    })
  })

  it('ends action highlighting exactly when the camera segment ends', () => {
    expect(cameraSegmentIsActive(0, 3)).toBe(true)
    expect(cameraSegmentIsActive(2.999, 3)).toBe(true)
    expect(cameraSegmentIsActive(3, 3)).toBe(false)
    expect(cameraSegmentIsActive(3.05, 3)).toBe(false)
  })

  it('shows MOVE/LOOK keys and a translucent path map and supports hiding them', async () => {
    const onVisibilityToggle = vi.fn()
    const path: CameraPath = {
      id: 'camera-test',
      name: 'Custom path 22:28',
      source: 'recorded',
      trajectory: '{}',
      durationMs: 1000,
      sampleRate: 24,
      frames: [{
        frame: 0,
        t: 0,
        position: { x: 0, y: 0, z: 0 },
        quaternion: { x: 0, y: 0, z: 0, w: 1 },
        fov: 55,
      }],
      keyEvents: [],
    }
    const videoRef = createRef<HTMLVideoElement>()
    const { rerender } = render(createElement(CameraPlaybackOverlay, {
      path,
      videoRef,
      segmentStartSeconds: 0,
      segmentDurationSeconds: 1,
      visible: true,
      onVisibilityToggle,
    }))

    expect(screen.getByLabelText('Camera movement')).toHaveTextContent('WASDMOVE')
    expect(screen.getByLabelText('Camera rotation')).toHaveTextContent('↑←↓→LOOK')
    expect(screen.getByLabelText('Camera path visualization')).toBeInTheDocument()
    expect(screen.getByText('Custom path 22:28')).toBeInTheDocument()
    const visibilityToggle = screen.getByRole('button', {
      name: 'Hide camera path',
    })
    expect(visibilityToggle).toHaveTextContent('Camera overlay on')
    expect(visibilityToggle).toHaveAttribute('aria-pressed', 'true')
    await userEvent.setup().click(
      screen.getByRole('button', { name: 'Hide camera path' }),
    )
    expect(onVisibilityToggle).toHaveBeenCalledOnce()

    rerender(createElement(CameraPlaybackOverlay, {
      path,
      videoRef,
      segmentStartSeconds: 0,
      segmentDurationSeconds: 1,
      visible: false,
      onVisibilityToggle,
    }))
    expect(screen.queryByLabelText('Camera path visualization')).not.toBeInTheDocument()
    const hiddenVisibilityToggle = screen.getByRole('button', {
      name: 'Show camera path',
    })
    expect(hiddenVisibilityToggle).toHaveTextContent('Camera overlay off')
    expect(hiddenVisibilityToggle).toHaveAttribute('aria-pressed', 'false')
    await userEvent.setup().click(
      screen.getByRole('button', { name: 'Show camera path' }),
    )
    expect(onVisibilityToggle).toHaveBeenCalledTimes(2)
  })
})

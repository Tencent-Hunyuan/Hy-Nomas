import { useEffect, useRef, useState } from 'react'
import type { CameraPath } from '../types'
import { StructuredPromptView } from './StructuredPromptView'

export type PromptTurnStage =
  | 'simulating'
  | 'rendering'
  | 'playing'
  | 'complete'
  | 'failed'

export interface PromptTurn {
  id: number
  number: number
  referenceImage?: boolean
  prompt: string
  submittedAt: string
  stage: PromptTurnStage
  k3Reasoning?: string
  k3Output?: string
  k3OutputComplete?: boolean
  structuredPrompt?: string
  requestId?: string
  promptEnhancementMode?: string
  promptEnhancerSeed?: number
  frames?: number
  fps?: number
  generationSeconds?: number
  startSeconds?: number
  endSeconds?: number
  playbackProgress?: number
  cameraLabel?: string
  cameraPath?: CameraPath
  error?: string
}

const stageLabels = ['Prompt', 'Simulating', 'Rendering', 'Playing', 'Complete']

const stageIndexes: Record<PromptTurnStage, number> = {
  simulating: 1,
  rendering: 2,
  playing: 3,
  complete: 4,
  failed: 4,
}

const statusLabels: Record<PromptTurnStage, string> = {
  simulating: 'Simulating',
  rendering: 'Rendering',
  playing: 'Playing',
  complete: 'Complete',
  failed: 'Failed',
}

function formatTime(seconds = 0) {
  const minutes = Math.floor(Math.max(0, seconds) / 60)
  const rest = Math.max(0, seconds) - minutes * 60
  return `${String(minutes).padStart(2, '0')}:${rest.toFixed(1).padStart(4, '0')}`
}

interface PromptTurnTimelineProps {
  turns: PromptTurn[]
}

export function PromptTurnTimeline({ turns }: PromptTurnTimelineProps) {
  const [expandedTurn, setExpandedTurn] = useState<number>()
  const activeRef = useRef<HTMLElement>(null)
  const activeOutputRef = useRef<HTMLDivElement>(null)

  const activeTurn =
    [...turns].reverse().find((turn) => turn.stage !== 'complete')
    ?? turns.at(-1)

  useEffect(() => {
    const active = activeRef.current
    const scroller = active?.closest('.message-list')
    if (!(active instanceof HTMLElement) || !(scroller instanceof HTMLElement)) return
    const top =
      active.offsetTop
      - scroller.offsetTop
      - Math.max(12, (scroller.clientHeight - active.offsetHeight) / 2)
    const target = Math.max(0, top)
    if (typeof scroller.scrollTo === 'function') {
      scroller.scrollTo({ top: target, behavior: 'smooth' })
    } else {
      scroller.scrollTop = target
    }
  }, [activeTurn?.id, activeTurn?.stage])

  useEffect(() => {
    const output = activeOutputRef.current
    if (output) {
      output.scrollTop = activeTurn?.k3OutputComplete
        ? 0
        : output.scrollHeight
    }
  }, [
    activeTurn?.k3Output,
    activeTurn?.k3OutputComplete,
    activeTurn?.k3Reasoning,
  ])

  if (turns.length === 0) {
    return (
      <div className="prompt-timeline-empty">
        <strong>Your world is ready.</strong>
        <span>Enter the first prompt to begin the timeline.</span>
      </div>
    )
  }

  return (
    <div className="prompt-turn-timeline">
      {turns.map((turn) => {
        const active = activeTurn?.id === turn.id
        const expanded = active || expandedTurn === turn.id
        const stageIndex = stageIndexes[turn.stage]
        const k3Output = turn.k3Output ?? turn.structuredPrompt
        const k3Reasoning = k3Output ? undefined : turn.k3Reasoning
        const duration =
          turn.startSeconds !== undefined && turn.endSeconds !== undefined
            ? Math.max(0, turn.endSeconds - turn.startSeconds)
            : undefined
        return (
          <article
            className={[
              'prompt-turn-card',
              active ? 'active' : '',
              turn.stage === 'complete' ? 'complete' : '',
              turn.stage === 'failed' ? 'failed' : '',
              expanded ? 'expanded' : '',
            ].filter(Boolean).join(' ')}
            key={turn.id}
            ref={active ? activeRef : undefined}
            onClick={() => setExpandedTurn((current) => current === turn.id ? undefined : turn.id)}
          >
            <header className="prompt-turn-head">
              <div>
                <strong>{turn.referenceImage ? 'T0-Reference Image' : `T${turn.number} · ${turn.cameraPath ? 'CAMERA' : 'TEXT'}`}</strong>
                <time>{turn.submittedAt}</time>
              </div>
              <span>{statusLabels[turn.stage]}</span>
            </header>

            <small className="prompt-turn-label">User Prompt</small>
            <p className="prompt-turn-copy">{turn.prompt}</p>

            {expanded && (
              <>
                <div className="prompt-turn-metrics">
                  {turn.frames !== undefined && <span>{turn.frames} frames</span>}
                  {duration !== undefined && <span>{duration.toFixed(1)}s video</span>}
                  {turn.generationSeconds !== undefined && (
                    <span>{turn.generationSeconds.toFixed(1)}s latency</span>
                  )}
                  {turn.promptEnhancementMode && (
                    <span>
                      {turn.promptEnhancementMode}
                    </span>
                  )}
                  {turn.promptEnhancerSeed !== undefined && (
                    <span>LLM seed {turn.promptEnhancerSeed}</span>
                  )}
                  {turn.cameraLabel && <span>{turn.cameraLabel}</span>}
                </div>

                {(k3Output || k3Reasoning || turn.stage === 'simulating') && (
                  <div
                    className={[
                      'prompt-turn-k3',
                      k3Output ? 'content' : 'reasoning',
                      turn.k3OutputComplete ? 'complete' : 'streaming',
                    ].filter(Boolean).join(' ')}
                    aria-label={`Enhanced output for prompt ${turn.number}`}
                    ref={active ? activeOutputRef : undefined}
                  >
                    <header>
                      <b>
                        {k3Output
                          ? turn.k3OutputComplete
                            ? 'Enhanced Scene Plan'
                            : 'Enhanced Scene Plan · Streaming'
                          : 'Enhanced Thinking'}
                      </b>
                      {!turn.k3OutputComplete && <i aria-hidden="true" />}
                    </header>
                    {k3Output ? (
                      <StructuredPromptView
                        source={k3Output}
                        complete={Boolean(turn.k3OutputComplete)}
                      />
                    ) : (
                      <p>
                        {k3Reasoning
                          || 'Preparing the structured segment prompt…'}
                      </p>
                    )}
                  </div>
                )}
              </>
            )}

            <div className="prompt-turn-stages" aria-label={`Prompt ${turn.number} status`}>
              {stageLabels.map((label, index) => (
                <span
                  className={[
                    index < stageIndex ? 'done' : '',
                    index === stageIndex ? 'current' : '',
                  ].filter(Boolean).join(' ')}
                  key={label}
                >
                  {label}
                </span>
              ))}
            </div>

            <div className="prompt-turn-progress" aria-hidden="true">
              <i style={{ width: `${Math.max(0, Math.min(100, turn.playbackProgress ?? 0))}%` }} />
            </div>

            <footer className="prompt-turn-footer">
              <span>{turn.requestId ?? 'Preparing request'}</span>
              <span>
                {turn.startSeconds !== undefined && turn.endSeconds !== undefined
                  ? `${formatTime(turn.startSeconds)} → ${formatTime(turn.endSeconds)}`
                  : statusLabels[turn.stage]}
              </span>
            </footer>

            {turn.error && <p className="prompt-turn-error">{turn.error}</p>}
          </article>
        )
      })}
    </div>
  )
}

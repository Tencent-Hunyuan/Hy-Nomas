import type { PromptTurn } from './PromptTurnTimeline'

type TimedTurn = PromptTurn & { startSeconds: number, endSeconds: number }

export function DemoOverlayTimeline({
  turns, currentTime,
}: { turns: PromptTurn[], currentTime: number }) {
  const timed = turns.filter((turn): turn is TimedTurn => (
    turn.startSeconds !== undefined && turn.endSeconds !== undefined
    && Number.isFinite(turn.startSeconds) && Number.isFinite(turn.endSeconds)
    && turn.endSeconds > turn.startSeconds
  ))
  if (!timed.length) return null
  return (
    <div className="demo-overlay-timeline" aria-label="Video turn timeline">
      {timed.map((turn, index) => {
        const duration = turn.endSeconds - turn.startSeconds
        const progress = Math.max(0, Math.min(1, (currentTime - turn.startSeconds) / duration))
        const active = currentTime >= turn.startSeconds && (
          currentTime < turn.endSeconds
          || (index === timed.length - 1 && currentTime === turn.endSeconds)
        )
        return (
          <div className="demo-timeline-segment" key={turn.id} style={{ flexGrow: duration }}>
            <span>T{turn.number} · {turn.cameraPath ? 'CAMERA' : index === 0 ? 'START' : 'TEXT'}</span>
            <div className="demo-timeline-track">
              <i style={{ width: `${progress * 100}%` }}>
                {active && <b />}
              </i>
            </div>
          </div>
        )
      })}
    </div>
  )
}

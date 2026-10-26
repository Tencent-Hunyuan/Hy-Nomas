import { OverlayVisibilityToggle } from './OverlayVisibilityToggle'

interface DirectionHudProps {
  direction: string
  visible: boolean
  onVisibilityToggle: () => void
}

export function DirectionHud({
  direction,
  visible,
  onVisibilityToggle,
}: DirectionHudProps) {
  return (
    <>
      <OverlayVisibilityToggle
        kind="prompt"
        visible={visible}
        onToggle={onVisibilityToggle}
      />
      {visible && (
        <aside className="demo-prompt-overlay" aria-label="Real user prompt">
          <span className="demo-text-badge">TEXT</span>
          <p>{direction}</p>
        </aside>
      )}
    </>
  )
}

import { Eye, EyeOff } from 'lucide-react'

interface OverlayVisibilityToggleProps {
  kind: 'camera' | 'prompt'
  visible: boolean
  onToggle: () => void
}

export function OverlayVisibilityToggle({
  kind,
  visible,
  onToggle,
}: OverlayVisibilityToggleProps) {
  const label = kind === 'camera' ? 'Camera' : 'Prompt'
  const target = kind === 'camera' ? 'camera path' : 'real user prompt'

  return (
    <button
      className={`demo-overlay-toggle demo-overlay-reveal ${visible ? 'is-on' : 'is-off'}`}
      type="button"
      aria-label={`${visible ? 'Hide' : 'Show'} ${target}`}
      aria-pressed={visible}
      onClick={onToggle}
    >
      {visible ? <Eye size={17} /> : <EyeOff size={17} />}
      <span>{label} overlay {visible ? 'on' : 'off'}</span>
    </button>
  )
}

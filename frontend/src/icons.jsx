const STROKE = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round', strokeLinejoin: 'round' }

export function IconLeaf({ size = 24 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...STROKE}>
      <path d="M5 21c.5-3 1.5-5 4-7s5-3 9-3c0 4-1 7-3 9s-4.5 3-7.5 3c-1 0-2 0-2.5-2" />
      <path d="M5 21c4-4 7-7 10-10" />
    </svg>
  )
}

export function IconCloud({ size = 24 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...STROKE}>
      <path d="M7 16a4 4 0 1 1 .5-7.9 5 5 0 0 1 9.7 1.4A3.5 3.5 0 0 1 17 16H7Z" />
      <path d="M8 19l-1 2M12 19l-1 2M16 19l-1 2" />
    </svg>
  )
}

export function IconBars({ size = 24 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...STROKE}>
      <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" />
    </svg>
  )
}

export function IconCalendar({ size = 24 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...STROKE}>
      <rect x="3" y="5" width="18" height="16" rx="2" />
      <path d="M8 3v4M16 3v4M3 10h18" />
      <path d="M8 14h2M14 14h2M8 18h2M14 18h2" />
    </svg>
  )
}

export function IconArrow({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...STROKE}>
      <path d="M5 12h14M13 5l7 7-7 7" />
    </svg>
  )
}

export function IconPlus({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...STROKE}>
      <path d="M12 5v14M5 12h14" />
    </svg>
  )
}

export function IconTrash({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...STROKE}>
      <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
    </svg>
  )
}

export function IconEdit({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...STROKE}>
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
      <path d="M18.5 2.5a2.121 2.121 0 1 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
    </svg>
  )
}

export function IconHome({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...STROKE}>
      <path d="M3 11l9-8 9 8" />
      <path d="M5 10v10h14V10" />
    </svg>
  )
}

export function CoffeeBeanMascot({ size = 260 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 260 260" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="bean" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#8A6648" />
          <stop offset="100%" stopColor="#5C3A1E" />
        </linearGradient>
      </defs>
      <ellipse cx="130" cy="245" rx="80" ry="6" fill="rgba(19,33,46,0.12)" />
      <path d="M 130 30 C 75 30, 50 90, 60 150 C 70 215, 110 240, 130 240 C 150 240, 190 215, 200 150 C 210 90, 185 30, 130 30 Z" fill="url(#bean)" />
      <path d="M 130 35 Q 90 130, 130 235 Q 170 130, 130 35 Z" fill="none" stroke="#3a2412" strokeWidth="3" />
      <circle cx="105" cy="135" r="8" fill="#FFF6D6" />
      <circle cx="155" cy="135" r="8" fill="#FFF6D6" />
      <circle cx="107" cy="137" r="3" fill="#13212E" />
      <circle cx="157" cy="137" r="3" fill="#13212E" />
      <path d="M 110 165 Q 130 180, 150 165" fill="none" stroke="#FFF6D6" strokeWidth="3" strokeLinecap="round" />
      <circle cx="85" cy="155" r="6" fill="#FF8A6B" opacity="0.55" />
      <circle cx="175" cy="155" r="6" fill="#FF8A6B" opacity="0.55" />
      <path d="M 80 90 Q 60 60, 50 70 Q 55 90, 75 100 Z" fill="#3F8A3E" />
      <path d="M 180 90 Q 200 60, 210 70 Q 205 90, 185 100 Z" fill="#3F8A3E" />
    </svg>
  )
}

export function IconMenu({ size = 22 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...STROKE}>
      <path d="M3 6h18M3 12h18M3 18h18" />
    </svg>
  )
}

export function IconX({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...STROKE}>
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  )
}

export function MountainsBg() {
  return (
    <svg viewBox="0 0 1440 240" preserveAspectRatio="none" style={{ position: 'absolute', bottom: 0, left: 0, width: '100%', height: 200, opacity: 0.4, pointerEvents: 'none' }}>
      <path d="M0 240 L0 180 L160 120 L300 160 L440 80 L600 140 L780 60 L940 130 L1100 90 L1280 150 L1440 110 L1440 240 Z" fill="#3F7BB8" opacity="0.45" />
      <path d="M0 240 L0 210 L120 160 L280 190 L420 140 L580 180 L740 130 L900 175 L1060 145 L1220 180 L1440 160 L1440 240 Z" fill="#1B4A6B" opacity="0.55" />
    </svg>
  )
}

const STORAGE_KEY = 'coffeemind.parcelas.v2'

export const DEFAULT_PARCELAS = [
  {
    id: 'p-bajio',
    nombre: 'Bajío Caliente',
    region: 'Costa Sur',
    altitud: 900,
    ph: 5.0,
    sombra: 0,
    N: 100,
    P: 30,
    K: 30,
    temperature: 27,
    humidity: 55,
    rainfall: 600,
  },
  {
    id: 'p-alpino',
    nombre: 'Alpino Frío',
    region: 'Cuchumatanes',
    altitud: 2000,
    ph: 5.8,
    sombra: 2,
    N: 90,
    P: 35,
    K: 35,
    temperature: 20,
    humidity: 75,
    rainfall: 1100,
  },
]

export function loadParcelas() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULT_PARCELAS
    const parsed = JSON.parse(raw)
    if (Array.isArray(parsed) && parsed.length) return parsed
    return DEFAULT_PARCELAS
  } catch {
    return DEFAULT_PARCELAS
  }
}

export function saveParcelas(parcelas) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(parcelas))
  } catch {
    // localStorage unavailable
  }
}

export function newParcelaId() {
  return `p-${Math.random().toString(36).slice(2, 8)}`
}

export const SHADE_LABELS = ['sol pleno', 'semi-sombra', 'sombra densa']

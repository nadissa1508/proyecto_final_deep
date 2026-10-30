// Regiones con pronóstico (una estación de ANACAFE cada una). Las claves coinciden con configs/regions.yaml del backend.
export const REGIONS = [
  { key: 'huehuetenango', label: 'Huehuetenango' },
  { key: 'antigua', label: 'Antigua Guatemala' },
  { key: 'coban', label: 'Cobán' },
  { key: 'atitlan', label: 'Atitlán (Nahualá)' },
  { key: 'fraijanes', label: 'Fraijanes' },
  { key: 'nuevo_oriente', label: 'Nuevo Oriente' },
]

const STORAGE_KEY = 'coffeemind.weatherRegion.v1'

const normalize = (s) => (s || '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '')

/** Región guardada para la parcela o, si el nombre de su región coincide con alguna, esa; si no, la primera. */
export function guessRegion(parcela) {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')[parcela.id]
    if (saved) return saved
  } catch { /* sin almacenamiento: se usa la suposición */ }
  const text = normalize(parcela.region)
  const hit = REGIONS.find((r) => text.includes(normalize(r.label).split(' ')[0]))
  return hit ? hit.key : REGIONS[0].key
}

export function saveRegion(parcelaId, key) {
  try {
    const all = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...all, [parcelaId]: key }))
  } catch { /* sin almacenamiento: no se recuerda */ }
}

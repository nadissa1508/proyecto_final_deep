const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

async function request(path, { method = 'GET', body, form } = {}) {
  const init = { method }
  if (form !== undefined) {
    init.body = form // FormData: el navegador pone el Content-Type con su separador
  } else if (body !== undefined) {
    init.headers = { 'Content-Type': 'application/json' }
    init.body = JSON.stringify(body)
  }
  let res
  try {
    res = await fetch(`${API_BASE}${path}`, init)
  } catch (e) {
    throw new Error(`No se pudo conectar al backend en ${API_BASE}. ¿Está corriendo?`)
  }
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    let detail = text
    try { detail = JSON.parse(text).detail ?? text } catch { /* texto plano */ }
    throw new Error(`${res.status} ${res.statusText}${detail ? ': ' + (typeof detail === 'string' ? detail : JSON.stringify(detail)) : ''}`)
  }
  return res.json()
}

export const cspSolve = (payload) => request('/api/csp/solve', { method: 'POST', body: payload })
export const bayesianInfer = (payload) => request('/api/bayesian/infer', { method: 'POST', body: payload })
export const mlPredict = (payload) => request('/api/ml/predict', { method: 'POST', body: payload })
export const mdpPolicy = (parcela) =>
  parcela
    ? request('/api/mdp/policy', { method: 'POST', body: parcela })
    : request('/api/mdp/policy')
export const health = () => request('/health')

// Deep Learning (Fase 4)
export const forecastGet = (region, source = 'live') => request(`/api/forecast/${region}?source=${source}`)

export function visionPredict(files, { heatmap = false } = {}) {
  const form = new FormData()
  files.forEach((f) => form.append('files', f, f.name))
  return request(`/api/vision/predict?heatmap=${heatmap}`, { method: 'POST', form })
}

export function decide({ region, files = [], altitud, sombra, humedadTipica, source = 'live' }) {
  const form = new FormData()
  form.append('region', region)
  form.append('source', source)
  if (altitud != null) form.append('altitud', String(altitud))
  if (sombra != null) form.append('sombra', String(sombra))
  if (humedadTipica != null) form.append('humedad_tipica', String(humedadTipica))
  files.forEach((f) => form.append('files', f, f.name))
  return request('/api/decision', { method: 'POST', form })
}

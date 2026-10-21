const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

async function request(path, { method = 'GET', body } = {}) {
  const init = { method }
  if (body !== undefined) {
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
    throw new Error(`${res.status} ${res.statusText}${text ? ': ' + text : ''}`)
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

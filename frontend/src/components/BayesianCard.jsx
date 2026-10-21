import { useEffect, useState } from 'react'
import { bayesianInfer } from '../api'
import { IconCloud } from '../icons'

function BarTriple({ data, kinds }) {
  const labels = ['bajo', 'medio', 'alto']
  return labels.map((label, i) => (
    <div className="bar-row" key={label}>
      <span style={{ textTransform: 'capitalize' }}>{label}</span>
      <div className="bar-track">
        <div className={`bar-fill bar-fill--${kinds[i]}`} style={{ width: `${(data[label] || 0) * 100}%` }} />
      </div>
      <span className="num">{((data[label] || 0) * 100).toFixed(1)}%</span>
    </div>
  ))
}

export default function BayesianCard({ parcela }) {
  const [state, setState] = useState({ loading: true, data: null, error: null })

  useEffect(() => {
    let cancelled = false
    setState({ loading: true, data: null, error: null })
    bayesianInfer({
      temperatura: parcela.temperature,
      precipitacion: parcela.rainfall,
      humedad: parcela.humidity,
    })
      .then((data) => !cancelled && setState({ loading: false, data, error: null }))
      .catch((err) => !cancelled && setState({ loading: false, data: null, error: err.message }))
    return () => { cancelled = true }
  }, [parcela.temperature, parcela.rainfall, parcela.humidity])

  const r = state.data?.riesgo_roya
  const y = state.data?.rendimiento

  const dominantRiesgo = r ? Object.entries(r).sort((a, b) => b[1] - a[1])[0] : null
  const chipKind = dominantRiesgo
    ? (dominantRiesgo[0] === 'alto' ? 'chip--risk' : dominantRiesgo[0] === 'medio' ? 'chip--warn' : 'chip--ok')
    : 'chip--neutral'

  return (
    <article className="module-card module-card--bayesian">
      <div className="module-card__head">
        <div>
          <div className="module-card__title">Riesgo climático</div>
          <div className="module-card__subtitle">Red bayesiana · Variable Elimination</div>
        </div>
        <div className="card__icon card__icon--mint" style={{ marginBottom: 0 }}><IconCloud /></div>
      </div>

      {state.loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <div>
            <div className="skeleton skeleton-line skeleton-line--sm" style={{ width: '50%' }} />
            <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'baseline', marginTop: '0.25rem' }}>
              <div className="skeleton skeleton-line skeleton-line--lg" style={{ width: '38%' }} />
              <div className="skeleton skeleton-chip" style={{ width: 46 }} />
            </div>
          </div>
          {['Riesgo de roya', 'Rendimiento'].map((label) => (
            <div key={label}>
              <div className="skeleton skeleton-line skeleton-line--sm" style={{ width: '42%', marginBottom: '0.5rem' }} />
              {[70, 90, 55].map((w) => (
                <div key={w} className="bar-row" style={{ marginBottom: '0.35rem' }}>
                  <div className="skeleton" style={{ width: 38, height: 12, borderRadius: 4 }} />
                  <div className="skeleton skeleton-bar" />
                  <div className="skeleton" style={{ width: 34, height: 12, borderRadius: 4 }} />
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
      {state.error && <div className="error">{state.error}</div>}

      {r && y && (
        <>
          <div>
            <div className="kpi__label">Riesgo de roya dominante</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem', marginTop: '0.25rem' }}>
              <div className="module-card__hero" style={{ textTransform: 'capitalize' }}>{dominantRiesgo[0]}</div>
              <span className={`chip ${chipKind}`}>{(dominantRiesgo[1] * 100).toFixed(0)}%</span>
            </div>
          </div>

          <div>
            <div className="kpi__label" style={{ marginBottom: '0.5rem' }}>P(Riesgo de roya)</div>
            <BarTriple data={r} kinds={['ok', 'warn', 'risk']} />
          </div>

          <div>
            <div className="kpi__label" style={{ marginBottom: '0.5rem' }}>P(Rendimiento)</div>
            <BarTriple data={y} kinds={['risk', 'warn', 'ok']} />
          </div>
        </>
      )}
    </article>
  )
}

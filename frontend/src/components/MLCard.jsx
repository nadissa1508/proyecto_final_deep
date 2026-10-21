import { useEffect, useState } from 'react'
import { mlPredict } from '../api'
import { IconBars } from '../icons'

export default function MLCard({ parcela }) {
  const [state, setState] = useState({ loading: true, data: null, error: null })

  useEffect(() => {
    let cancelled = false
    setState({ loading: true, data: null, error: null })
    mlPredict({
      N: parcela.N, P: parcela.P, K: parcela.K,
      temperature: parcela.temperature,
      humidity: parcela.humidity,
      ph: parcela.ph,
      rainfall: parcela.rainfall,
    })
      .then((data) => !cancelled && setState({ loading: false, data, error: null }))
      .catch((err) => !cancelled && setState({ loading: false, data: null, error: err.message }))
    return () => { cancelled = true }
  }, [parcela.N, parcela.P, parcela.K, parcela.temperature, parcela.humidity, parcela.ph, parcela.rainfall])

  const probs = state.data?.probabilidades
  const top = state.data?.cultivo_recomendado
  const isCoffee = top === 'coffee'

  return (
    <article className="module-card module-card--ml">
      <div className="module-card__head">
        <div>
          <div className="module-card__title">Cultivo más apto</div>
          <div className="module-card__subtitle">GaussianNB · implementación propia</div>
        </div>
        <div className="card__icon card__icon--cream" style={{ marginBottom: 0 }}><IconBars /></div>
      </div>

      {state.loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <div>
            <div className="skeleton skeleton-line skeleton-line--sm" style={{ width: '30%' }} />
            <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'baseline', marginTop: '0.25rem' }}>
              <div className="skeleton skeleton-line skeleton-line--lg" style={{ width: '44%' }} />
              <div className="skeleton skeleton-chip" style={{ width: 102 }} />
            </div>
          </div>
          <div>
            <div className="skeleton skeleton-line skeleton-line--sm" style={{ width: '40%', marginBottom: '0.5rem' }} />
            {[80, 100, 65, 55, 45].map((w) => (
              <div key={w} className="bar-row" style={{ marginBottom: '0.35rem' }}>
                <div className="skeleton" style={{ width: 48, height: 12, borderRadius: 4 }} />
                <div className="skeleton skeleton-bar" />
                <div className="skeleton" style={{ width: 34, height: 12, borderRadius: 4 }} />
              </div>
            ))}
          </div>
        </div>
      )}
      {state.error && <div className="error">{state.error}</div>}

      {top && (
        <>
          <div>
            <div className="kpi__label">Predicción</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem', marginTop: '0.25rem' }}>
              <div className="module-card__hero" style={{ textTransform: 'capitalize' }}>{top}</div>
              <span className={`chip ${isCoffee ? 'chip--ok' : 'chip--warn'}`}>
                {isCoffee ? 'apto para café' : 'sugiere otro cultivo'}
              </span>
            </div>
            {!isCoffee && (
              <div className="small muted" style={{ marginTop: '0.5rem' }}>
                Con estos parámetros el modelo entrenado en crop_recommendation.csv recomienda otro cultivo.
                Considerá ajustar NPK o el clima de referencia.
              </div>
            )}
          </div>

          {probs && (
            <div>
              <div className="kpi__label" style={{ marginBottom: '0.5rem' }}>Top probabilidades</div>
              {Object.entries(probs).map(([cls, p]) => (
                <div className="bar-row" key={cls}>
                  <span>{cls}</span>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${p * 100}%` }} />
                  </div>
                  <span className="num">{(p * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </article>
  )
}

import { Fragment, useEffect, useState } from 'react'
import { mdpPolicy } from '../api'
import { IconCalendar } from '../icons'

export default function MDPCard({ parcela }) {
  const [state, setState] = useState({ loading: true, data: null, error: null })

  useEffect(() => {
    let cancelled = false
    setState({ loading: true, data: null, error: null })
    mdpPolicy({
      altitud: parcela.altitud,
      sombra: parcela.sombra,
      humedad_tipica: parcela.humidity,
    })
      .then((data) => !cancelled && setState({ loading: false, data, error: null }))
      .catch((err) => !cancelled && setState({ loading: false, data: null, error: err.message }))
    return () => { cancelled = true }
  }, [parcela.altitud, parcela.sombra, parcela.humidity])

  const policy = state.data

  return (
    <article className="module-card module-card--mdp">
      <div className="module-card__head">
        <div>
          <div className="module-card__title">Plan estacional óptimo</div>
          <div className="module-card__subtitle">MDP · Value Iteration condicionado a la parcela</div>
        </div>
        <div className="card__icon card__icon--coral" style={{ marginBottom: 0 }}><IconCalendar /></div>
      </div>

      {state.loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <div className="skeleton skeleton-chip" style={{ width: 80 }} />
            <div className="skeleton skeleton-chip" style={{ width: 118 }} />
            <div className="skeleton skeleton-chip" style={{ width: 96 }} />
          </div>
          <div className="heatmap">
            {Array.from({ length: 4 * 13 }).map((_, i) => (
              <div key={i} className="skeleton" style={{ height: 48, borderRadius: 'var(--r-sm)' }} />
            ))}
          </div>
        </div>
      )}
      {state.error && <div className="error">{state.error}</div>}

      {policy && (
        <>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <span className="chip">{policy.iteraciones} iter VI</span>
            <span className="chip chip--mint">presión roya {policy.rust_pressure?.toFixed(2)}</span>
            <span className="chip chip--cream">precio ${policy.precio_qq?.toFixed(0)}/qq</span>
          </div>

          <div className="heatmap">
            <div className="heatmap__cell heatmap__cell--head">mes</div>
            {policy.niveles_roya.map((r) => (
              <div className="heatmap__cell heatmap__cell--head" key={r}>roya {r}</div>
            ))}
            {policy.meses.map((m, i) => (
              <Fragment key={m}>
                <div className="heatmap__cell heatmap__cell--row-head">{m}</div>
                {policy.politica[i].map((a, j) => (
                  <div key={j} className={`heatmap__cell action-${a}`}>
                    <span className="heatmap__action">{a}</span>
                    <span className="heatmap__value">V={Math.round(policy.valores[i][j])}</span>
                  </div>
                ))}
              </Fragment>
            ))}
          </div>
        </>
      )}
    </article>
  )
}

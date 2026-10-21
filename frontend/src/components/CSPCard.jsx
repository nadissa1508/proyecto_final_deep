import { useEffect, useState } from 'react'
import { cspSolve } from '../api'
import { IconLeaf } from '../icons'
import { SHADE_LABELS } from '../parcelas'

export default function CSPCard({ parcela }) {
  const [state, setState] = useState({ loading: true, data: null, error: null })

  useEffect(() => {
    let cancelled = false
    setState({ loading: true, data: null, error: null })
    cspSolve({
      parcelas: [{
        parcela_id: parcela.id,
        altitud: parcela.altitud,
        ph: parcela.ph,
        sombra: parcela.sombra,
      }],
      diversify: false,
      w_rust: 0.5,
      w_yield: 0.5,
    })
      .then((data) => !cancelled && setState({ loading: false, data, error: null }))
      .catch((err) => !cancelled && setState({ loading: false, data: null, error: err.message }))
    return () => { cancelled = true }
  }, [parcela.id, parcela.altitud, parcela.ph, parcela.sombra])

  const asignacion = state.data?.asignaciones?.[0]

  return (
    <article className="module-card module-card--csp">
      <div className="module-card__head">
        <div>
          <div className="module-card__title">Variedad recomendada</div>
          <div className="module-card__subtitle">CSP · Backtracking + FC + MRV</div>
        </div>
        <div className="card__icon card__icon--sky" style={{ marginBottom: 0 }}><IconLeaf /></div>
      </div>

      {state.loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <div className="skeleton skeleton-line skeleton-line--sm" style={{ width: '35%' }} />
          <div className="skeleton skeleton-line skeleton-line--lg" style={{ width: '65%' }} />
          <div className="skeleton skeleton-line skeleton-line--sm" style={{ width: '88%' }} />
          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.25rem' }}>
            <div className="skeleton skeleton-chip" style={{ width: 80 }} />
            <div className="skeleton skeleton-chip" style={{ width: 70 }} />
            <div className="skeleton skeleton-chip" style={{ width: 115 }} />
          </div>
        </div>
      )}
      {state.error && <div className="error">{state.error}</div>}

      {asignacion && (
        <>
          <div>
            <div className="kpi__label">Mejor variedad</div>
            <div className="module-card__hero">{asignacion.variedad}</div>
            <div className="muted small">
              Compatible con altitud {parcela.altitud} m, pH {parcela.ph} y sombra {SHADE_LABELS[parcela.sombra]}.
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <span className="chip chip--ok">score {asignacion.score.toFixed(2)}</span>
            <span className="chip">nodos {state.data.nodos_expandidos}</span>
            <span className="chip chip--neutral">restricciones duras OK</span>
          </div>
        </>
      )}
    </article>
  )
}

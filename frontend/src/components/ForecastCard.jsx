import { useEffect, useState } from 'react'
import { forecastGet } from '../api'
import { IconCloud } from '../icons'

const NAMES = {
  temperature_2m_mean: ['Temp. °C', 1],
  relative_humidity_2m_mean: ['Humedad %', 0],
  precipitation_sum: ['Lluvia mm', 1],
}

function dayLabel(iso) {
  const d = new Date(`${iso}T12:00:00`)
  return d.toLocaleDateString('es-GT', { weekday: 'short', day: 'numeric', month: 'short' })
}

export default function ForecastCard({ regionKey }) {
  const [state, setState] = useState({ loading: true, data: null, error: null })
  const [source, setSource] = useState('live')

  useEffect(() => {
    let cancelled = false
    setState({ loading: true, data: null, error: null })
    forecastGet(regionKey, source)
      .then((data) => !cancelled && setState({ loading: false, data, error: null }))
      .catch((err) => !cancelled && setState({ loading: false, data: null, error: err.message }))
    return () => { cancelled = true }
  }, [regionKey, source])

  const f = state.data

  return (
    <article className="module-card module-card--forecast">
      <div className="module-card__head">
        <div>
          <div className="module-card__title">Clima de los próximos 14 días</div>
          <div className="module-card__subtitle">GRU entrenado con ERA5 · afinado con estaciones de ANACAFE</div>
        </div>
        <div className="card__icon card__icon--mint" style={{ marginBottom: 0 }}><IconCloud /></div>
      </div>

      {state.loading && <div className="skeleton" style={{ height: 160, borderRadius: 'var(--r-md)' }} />}
      {state.error && <div className="error">{state.error}</div>}

      {f && (
        <>
          <div className="forecast__meta">
            <span className="chip chip--neutral">{f.fuente_historia}</span>
            <span className="chip chip--neutral">último dato: {f.ultimo_dia_observado}</span>
            {f.dias_de_retraso > 0 && <span className="chip chip--warn">{f.dias_de_retraso} días de retraso</span>}
            <label className="small muted forecast__source">
              <input type="checkbox" checked={source === 'local'} onChange={(e) => setSource(e.target.checked ? 'local' : 'live')} />
              {' '}usar datos descargados
            </label>
          </div>

          <div className="forecast__scroll" role="region" aria-label="Pronóstico diario" tabIndex={0}>
            <table className="forecast__table">
              <thead>
                <tr>
                  <th scope="col">Día</th>
                  {Object.values(NAMES).map(([label]) => <th scope="col" key={label} className="num">{label}</th>)}
                </tr>
              </thead>
              <tbody>
                {f.dias.map((iso, i) => (
                  <tr key={iso} className={f.poco_confiable[i] ? 'forecast__row--weak' : ''}>
                    <th scope="row">
                      {dayLabel(iso)}
                      {f.poco_confiable[i] && <span className="chip chip--warn forecast__flag" title="Poco confiable por el retraso de los datos">poco confiable</span>}
                    </th>
                    {Object.entries(NAMES).map(([key, [, digits]]) => (
                      <td className="num" key={key}>{f.valores[key][i].toFixed(digits)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {f.notas.map((n) => <div className="muted small" key={n}>{n}</div>)}
          <div className="muted small">
            El pronóstico mejora poco frente a la climatología a más de 7 días; use la tendencia, no el valor exacto.
          </div>
        </>
      )}
    </article>
  )
}

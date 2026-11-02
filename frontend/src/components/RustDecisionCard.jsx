import { useEffect, useRef, useState } from 'react'
import { decide, visionPredict } from '../api'
import { IconCalendar } from '../icons'
import LeafUploader, { MIN_RECOMMENDED } from './LeafUploader'

const LEVELS = ['bajo', 'medio', 'alto']
const KINDS = ['ok', 'warn', 'risk']
const ACTION_TEXT = {
  nada: 'No aplicar nada por ahora',
  fertilizar: 'Fertilizar',
  cobre: 'Fungicida de cobre (preventivo)',
  triazol: 'Fungicida triazol (curativo)',
  podar: 'Podar',
}
const LEAF_TEXT = { sana: 'sanas', roya_leve: 'con roya leve', roya_severa: 'con roya severa', otra_afeccion: 'con otra afección' }

function Bars({ data }) {
  return LEVELS.map((label, i) => (
    <div className="bar-row" key={label}>
      <span style={{ textTransform: 'capitalize' }}>{label}</span>
      <div className="bar-track">
        <div className={`bar-fill bar-fill--${KINDS[i]}`} style={{ width: `${(data[label] || 0) * 100}%` }} />
      </div>
      <span className="num">{((data[label] || 0) * 100).toFixed(0)}%</span>
    </div>
  ))
}

const pct = (x) => `${(x * 100).toFixed(0)}%`

const ALERT_TITLE = { alerta: 'Alerta de roya', sospecha: 'Posible roya', vigilancia: 'Vigilancia', seguimiento: 'Seguimiento normal' }
const ALERT_CHIP = { alerta: 'chip--warn', sospecha: 'chip--warn', vigilancia: 'chip--warn', seguimiento: 'chip--ok' }
const RUST_TEXT = { detectada: 'detectada', sospecha: 'sospecha (1 hoja)', no_detectada: 'no detectada', sin_fotos: 'sin fotos' }

// asOf viene de la tarjeta de pronóstico: vacía = hoy; con fecha = consulta histórica (uso académico)
export default function RustDecisionCard({ parcela, regionKey, asOf = '' }) {
  const [items, setItems] = useState([])
  const [state, setState] = useState({ loading: false, data: null, error: null })
  const [maps, setMaps] = useState({ loading: false, data: null, error: null })
  // Cada consulta lleva un número de versión: si cambia la parcela, la región o las fotos mientras espera, su respuesta tardía se ignora
  const requestVersion = useRef(0)
  const mapVersion = useRef(0)

  // otra parcela: las fotos de la anterior no le corresponden
  useEffect(() => {
    setItems((previous) => {
      previous.forEach((item) => URL.revokeObjectURL(item.url))
      return []
    })
  }, [parcela.id])

  // otra parcela, región, altitud, sombra o fecha de consulta: el resultado anterior ya no corresponde
  useEffect(() => {
    requestVersion.current += 1
    mapVersion.current += 1
    setState({ loading: false, data: null, error: null })
    setMaps({ loading: false, data: null, error: null })
  }, [parcela.id, parcela.altitud, parcela.sombra, regionKey, asOf])

  // otras fotos: la decisión y los mapas anteriores ya no corresponden
  const updateItems = (next) => {
    requestVersion.current += 1
    mapVersion.current += 1
    setItems(next)
    setState({ loading: false, data: null, error: null })
    setMaps({ loading: false, data: null, error: null })
  }

  const showMaps = () => {
    const version = ++mapVersion.current
    setMaps({ loading: true, data: null, error: null })
    visionPredict(items.map((x) => x.file), { heatmap: true })
      .then((data) => { if (version === mapVersion.current) setMaps({ loading: false, data, error: null }) })
      .catch((err) => { if (version === mapVersion.current) setMaps({ loading: false, data: null, error: err.message }) })
  }

  const run = () => {
    const version = ++requestVersion.current
    mapVersion.current += 1
    setMaps({ loading: false, data: null, error: null })
    setState({ loading: true, data: null, error: null })
    decide({ region: regionKey, files: items.map((x) => x.file), altitud: parcela.altitud, sombra: parcela.sombra, asOf: asOf || undefined })
      .then((data) => { if (version === requestVersion.current) setState({ loading: false, data, error: null }) })
      .catch((err) => { if (version === requestVersion.current) setState({ loading: false, data: null, error: err.message }) })
  }

  const d = state.data
  const counts = d ? d.hojas.reduce((acc, h) => ({ ...acc, [h]: (acc[h] || 0) + 1 }), {}) : {}

  return (
    <article className="module-card module-card--decision">
      <div className="module-card__head">
        <div>
          <div className="module-card__title">Decisión con fotos y pronóstico</div>
          <div className="module-card__subtitle">CNN + red bayesiana + MDP (QMDP)</div>
        </div>
        <div className="card__icon card__icon--coral" style={{ marginBottom: 0 }}><IconCalendar /></div>
      </div>

      <LeafUploader items={items} onChange={updateItems} disabled={state.loading} />

      {asOf && <div className="muted small">Consulta histórica desde el {asOf} (fecha elegida en la tarjeta de pronóstico).</div>}

      <div>
        <button type="button" className="btn btn--primary" onClick={run} disabled={state.loading}>
          {state.loading ? 'Analizando…' : items.length ? `Analizar ${items.length} ${items.length === 1 ? 'hoja' : 'hojas'}` : 'Decidir solo con el clima'}
        </button>
      </div>

      {state.error && <div className="error">{state.error}</div>}

      {d && (
        <>
          {d.consulta_historica && (
            <span className="chip chip--warn">Consulta histórica desde el {d.fecha_consulta}</span>
          )}
          <div>
            <div className="kpi__label">Acción recomendada este mes</div>
            <div className="module-card__hero">{ACTION_TEXT[d.accion] || d.accion}</div>
          </div>

          {d.alarma && (
            <div>
              <div className="kpi__label">Alerta de roya y riesgo climático</div>
              <span className={`chip ${ALERT_CHIP[d.alarma.nivel] || ''}`}>{ALERT_TITLE[d.alarma.nivel] || d.alarma.nivel}</span>
              <div className="small" style={{ marginTop: '0.4rem' }}>{d.alarma.mensaje}</div>
              <div className="muted small">
                Roya en las fotos: <strong>{RUST_TEXT[d.alarma.roya] || d.alarma.roya}</strong> · Riesgo climático:{' '}
                <strong>{d.alarma.riesgo_climatico}</strong>{d.alarma.poco_confiable ? ' (orientativo)' : ''}
              </div>
              <div className="muted small">{d.alarma.razones.join(' · ')}.</div>
              {d.alarma.discrepancia_mdp && <div className="error small">{d.alarma.discrepancia_mdp}</div>}
            </div>
          )}

          <div>
            <div className="kpi__label" style={{ marginBottom: '0.5rem' }}>Nivel de roya de la parcela, solo con el clima</div>
            <Bars data={d.creencia_solo_clima} />
          </div>
          {d.incidencia && (
            <div>
              <div className="kpi__label" style={{ marginBottom: '0.5rem' }}>Nivel de roya, con las {d.incidencia.hojas} hojas</div>
              <Bars data={d.creencia} />
            </div>
          )}

          {d.incidencia && (
            <div className="callout">
              <strong>{d.incidencia.hojas_con_roya} de {d.incidencia.hojas} hojas</strong> con roya
              ({Object.entries(counts).map(([k, n]) => `${n} ${LEAF_TEXT[k] || k}`).join(', ')}).
              {d.incidencia.hojas >= MIN_RECOMMENDED ? (
                <> Incidencia estimada <strong>{pct(d.incidencia.corregida)}</strong>
                  {' '}(intervalo 95 %: {pct(d.incidencia.intervalo_95[0])}–{pct(d.incidencia.intervalo_95[1])}),
                  corregida por los errores del clasificador.</>
              ) : (
                <> Con {d.incidencia.hojas} {d.incidencia.hojas === 1 ? 'hoja' : 'hojas'} no hay base para estimar la
                  incidencia: solo se puede decir que la incidencia real está entre {pct(d.incidencia.intervalo_95[0])} y {pct(d.incidencia.intervalo_95[1])}{' '}
                  (95 %). Fotografíe al menos {MIN_RECOMMENDED} hojas de plantas distintas.</>
              )}
              <div className="muted small">
                Con pocas hojas el intervalo es muy ancho: no sirve para decidir si se supera el 5 % de incidencia,
                solo para saber si hay roya. Sensibilidad {pct(d.incidencia.sensibilidad)}, especificidad {pct(d.incidencia.especificidad)} en las pruebas.
              </div>
            </div>
          )}

          {items.length > 0 && (
            <div>
              <button type="button" className="btn btn--secondary btn--sm" onClick={showMaps} disabled={maps.loading}>
                {maps.loading ? 'Calculando mapas…' : 'Ver dónde miró el modelo'}
              </button>
              {maps.error && <div className="error">{maps.error}</div>}
              {maps.data && (
                <>
                  <div className="thumbs thumbs--maps">
                    {maps.data.hojas.map((h, i) => (
                      <figure className="map" key={h.archivo + i}>
                        <img src={`data:image/png;base64,${h.mapa_calor_png_base64}`} alt={`Mapa de atención de la hoja ${i + 1}: ${h.clase}`} />
                        <figcaption>{h.clase.replace('_', ' ')} · {pct(h.confianza)}</figcaption>
                      </figure>
                    ))}
                  </div>
                  <div className="muted small">
                    Las zonas cálidas son las que más pesaron en la decisión del modelo. Muestra que mira la hoja y no el fondo,
                    pero no marca pústulas individuales: no lo use como diagnóstico.
                  </div>
                </>
              )}
            </div>
          )}

          <div>
            <div className="kpi__label" style={{ marginBottom: '0.5rem' }}>
              Clima de los próximos {d.clima.ventana_dias} días respecto de lo habitual en la finca
            </div>
            <div className="muted small" style={{ marginBottom: '0.5rem' }}>
              Temperatura media {d.clima.temperatura_media.toFixed(1)} °C · humedad {d.clima.humedad_media.toFixed(0)} % ·
              lluvia {d.clima.lluvia_total_mm.toFixed(0)} mm
            </div>
            {Object.entries(d.clima.respecto_de_lo_habitual).map(([name, dist]) => (
              <div key={name} style={{ marginBottom: '0.35rem' }}>
                <span className="small" style={{ textTransform: 'capitalize' }}>{name}: </span>
                <span className="small num">
                  {LEVELS.map((l) => `${l} ${pct(dist[l])}`).join(' · ')}
                </span>
              </div>
            ))}
            {d.clima.poco_confiable && <span className="chip chip--warn">pronóstico poco confiable</span>}
          </div>

          {d.notas.map((n) => <div className="muted small" key={n}>{n}</div>)}
          <div className="muted small">{d.advertencia}</div>
        </>
      )}
    </article>
  )
}

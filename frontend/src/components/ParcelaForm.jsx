import { useState } from 'react'

export default function ParcelaForm({ initial, onSave, onCancel }) {
  const [p, setP] = useState(initial)
  const update = (k, v) => setP({ ...p, [k]: v })
  const num = (k) => (e) => update(k, Number(e.target.value))

  const submit = (e) => {
    e.preventDefault()
    onSave({
      ...p,
      altitud: Number(p.altitud),
      ph: Number(p.ph),
      sombra: Number(p.sombra),
      N: Number(p.N), P: Number(p.P), K: Number(p.K),
      temperature: Number(p.temperature),
      humidity: Number(p.humidity),
      rainfall: Number(p.rainfall),
    })
  }

  return (
    <div className="parcela-form-modal" onClick={onCancel}>
      <form className="parcela-form-modal__panel" onClick={(e) => e.stopPropagation()} onSubmit={submit}>
        <div className="parcela-form-modal__head">
          <h3>{initial.id ? 'Editar parcela' : 'Nueva parcela'}</h3>
          <button type="button" className="btn btn--ghost" onClick={onCancel}>cerrar</button>
        </div>

        <div className="parcela-form__group">
          <h4>Identificación</h4>
          <div className="form-grid">
            <div className="field">
              <label>Nombre</label>
              <input value={p.nombre} onChange={(e) => update('nombre', e.target.value)} required />
            </div>
            <div className="field">
              <label>Región</label>
              <input value={p.region} onChange={(e) => update('region', e.target.value)} />
            </div>
          </div>
        </div>

        <div className="parcela-form__group">
          <h4>Geografía y suelo</h4>
          <p>Insumos del CSP y la red bayesiana.</p>
          <div className="form-grid">
            <div className="field">
              <label>Altitud (m)</label>
              <input type="number" min="0" max="4000" step="50" value={p.altitud} onChange={num('altitud')} />
            </div>
            <div className="field">
              <label>pH del suelo</label>
              <input type="number" min="3" max="9" step="0.1" value={p.ph} onChange={num('ph')} />
            </div>
            <div className="field">
              <label>Sombra disponible</label>
              <select value={p.sombra} onChange={num('sombra')}>
                <option value="0">0 — sol pleno</option>
                <option value="1">1 — semi-sombra</option>
                <option value="2">2 — sombra densa</option>
              </select>
            </div>
          </div>
        </div>

        <div className="parcela-form__group">
          <h4>Fertilidad (NPK, kg/ha)</h4>
          <p>Insumos del modelo ML.</p>
          <div className="form-grid">
            <div className="field"><label>Nitrógeno</label><input type="number" min="0" value={p.N} onChange={num('N')} /></div>
            <div className="field"><label>Fósforo</label><input type="number" min="0" value={p.P} onChange={num('P')} /></div>
            <div className="field"><label>Potasio</label><input type="number" min="0" value={p.K} onChange={num('K')} /></div>
          </div>
        </div>

        <div className="parcela-form__group">
          <h4>Clima típico</h4>
          <p>Promedios anuales o estacionales. Alimentan ML y red bayesiana.</p>
          <div className="form-grid">
            <div className="field">
              <label>Temperatura (°C)</label>
              <input type="number" min="0" max="40" step="0.5" value={p.temperature} onChange={num('temperature')} />
            </div>
            <div className="field">
              <label>Humedad (%)</label>
              <input type="number" min="0" max="100" step="1" value={p.humidity} onChange={num('humidity')} />
            </div>
            <div className="field">
              <label>Lluvia (mm/temp.)</label>
              <input type="number" min="0" max="5000" step="50" value={p.rainfall} onChange={num('rainfall')} />
            </div>
          </div>
        </div>

        <div className="parcela-form__actions">
          <button type="button" className="btn btn--secondary" onClick={onCancel}>Cancelar</button>
          <button type="submit" className="btn btn--primary">Guardar parcela</button>
        </div>
      </form>
    </div>
  )
}

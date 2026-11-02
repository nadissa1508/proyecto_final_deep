import { useRef, useState } from 'react'
import { MAX_FILES, prepareImage } from '../imageUtils'

export const MIN_RECOMMENDED = 10

/**
 * Subida de hojas desde archivo (botón o arrastrar y soltar). La captura con la cámara del celular queda como trabajo futuro.
 * Es controlado: `items` es la lista de { id, file, url } ya reducidas a ~1024 px.
 */
export default function LeafUploader({ items, onChange, disabled }) {
  const fileInput = useRef(null)
  const [over, setOver] = useState(false)
  const [errors, setErrors] = useState([])
  const [busy, setBusy] = useState(false)

  const add = async (fileList) => {
    const files = Array.from(fileList || [])
    if (!files.length) return
    setBusy(true)
    const next = [...items]
    const found = []
    for (const file of files) {
      if (next.length >= MAX_FILES) {
        found.push(`Máximo ${MAX_FILES} hojas por análisis.`)
        break
      }
      try {
        const { file: prepared, url } = await prepareImage(file)
        next.push({ id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, file: prepared, url })
      } catch (e) {
        found.push(e.message)
      }
    }
    setErrors(found)
    onChange(next)
    setBusy(false)
  }

  const remove = (id) => {
    const gone = items.find((x) => x.id === id)
    if (gone) URL.revokeObjectURL(gone.url)
    onChange(items.filter((x) => x.id !== id))
  }

  const clear = () => {
    items.forEach((x) => URL.revokeObjectURL(x.url))
    onChange([])
    setErrors([])
  }

  const onDrop = (e) => {
    e.preventDefault()
    setOver(false)
    if (!disabled) add(e.dataTransfer.files)
  }

  const reset = (e) => { e.target.value = '' } // permite volver a elegir el mismo archivo

  return (
    <div className="uploader">
      <div
        className={`dropzone${over ? ' dropzone--over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setOver(true) }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
      >
        <p className="dropzone__title">Arrastre aquí fotos de hojas o elíjalas</p>
        <div className="dropzone__actions">
          <button type="button" className="btn btn--secondary btn--sm" disabled={disabled || busy}
                  onClick={() => fileInput.current?.click()}>
            Elegir archivos
          </button>
        </div>
        <input ref={fileInput} type="file" accept="image/jpeg,image/png,image/webp" multiple hidden
               onChange={(e) => { add(e.target.files); reset(e) }} />
        <ul className="dropzone__guide">
          <li>Una hoja completa por foto, por el <strong>envés</strong> (ahí salen las pústulas).</li>
          <li>Si puede, sobre un papel blanco y con buena luz, sin sombras fuertes.</li>
          <li>De plantas distintas: al menos {MIN_RECOMMENDED} hojas (ideal 20–30).</li>
        </ul>
      </div>

      {busy && <div className="muted small" role="status">Preparando fotos…</div>}
      {errors.map((m) => <div className="error" key={m}>{m}</div>)}

      {items.length > 0 && (
        <>
          <div className="uploader__bar">
            <span className={`chip ${items.length >= MIN_RECOMMENDED ? 'chip--ok' : 'chip--warn'}`}>
              {items.length} {items.length === 1 ? 'hoja' : 'hojas'}
            </span>
            {items.length < MIN_RECOMMENDED && (
              <span className="muted small">Se recomiendan al menos {MIN_RECOMMENDED}.</span>
            )}
            <button type="button" className="btn btn--ghost btn--sm" onClick={clear} disabled={disabled}>Quitar todas</button>
          </div>
          <div className="thumbs">
            {items.map((x, i) => (
              <div className="thumb" key={x.id}>
                <img src={x.url} alt={`Hoja ${i + 1}`} />
                <button type="button" className="thumb__remove" aria-label={`Quitar hoja ${i + 1}`}
                        onClick={() => remove(x.id)} disabled={disabled}>×</button>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

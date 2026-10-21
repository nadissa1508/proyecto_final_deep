import { useEffect, useMemo, useState } from 'react'
import { health } from './api'
import { DEFAULT_PARCELAS, SHADE_LABELS, loadParcelas, newParcelaId, saveParcelas } from './parcelas'
import { IconEdit, IconHome, IconMenu, IconPlus, IconTrash, IconX } from './icons'
import ParcelaForm from './components/ParcelaForm'
import CSPCard from './components/CSPCard'
import BayesianCard from './components/BayesianCard'
import MLCard from './components/MLCard'
import MDPCard from './components/MDPCard'

const EMPTY = {
  id: '', nombre: 'Nueva parcela', region: '',
  altitud: 1500, ph: 5.5, sombra: 1,
  N: 90, P: 30, K: 30,
  temperature: 22, humidity: 70, rainfall: 1500,
}

export default function Dashboard({ onBackToLanding }) {
  const [parcelas, setParcelas] = useState(loadParcelas)
  const [selectedId, setSelectedId] = useState(parcelas[0]?.id)
  const [editing, setEditing] = useState(null)
  const [backendStatus, setBackendStatus] = useState('checking')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [delConfirm, setDelConfirm] = useState(null)

  useEffect(() => { saveParcelas(parcelas) }, [parcelas])

  useEffect(() => {
    health().then(() => setBackendStatus('online')).catch(() => setBackendStatus('offline'))
  }, [])

  const selected = useMemo(
    () => parcelas.find((p) => p.id === selectedId) || parcelas[0],
    [parcelas, selectedId],
  )

  useEffect(() => {
    document.title = selected ? `${selected.nombre} — CoffeeMind GT` : 'CoffeeMind GT'
    return () => { document.title = 'CoffeeMind GT' }
  }, [selected])

  const addParcela = () => setEditing({ ...EMPTY, id: newParcelaId() })

  const saveParcela = (p) => {
    const exists = parcelas.some((x) => x.id === p.id)
    const next = exists
      ? parcelas.map((x) => (x.id === p.id ? p : x))
      : [...parcelas, p]
    setParcelas(next)
    setSelectedId(p.id)
    setEditing(null)
  }

  const removeParcela = (id) => {
    const next = parcelas.filter((p) => p.id !== id)
    setParcelas(next.length ? next : DEFAULT_PARCELAS)
    if (selectedId === id) setSelectedId((next[0] || DEFAULT_PARCELAS[0]).id)
  }

  return (
    <>
      <div
        className={`sidebar-backdrop${sidebarOpen ? ' sidebar-backdrop--visible' : ''}`}
        onClick={() => setSidebarOpen(false)}
      />

      <div className="app-shell">
        <header className="app-topbar">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button
              className="btn btn--hamburger"
              aria-label={sidebarOpen ? 'Cerrar menú de parcelas' : 'Abrir menú de parcelas'}
              onClick={() => setSidebarOpen((v) => !v)}
            >
              {sidebarOpen ? <IconX /> : <IconMenu />}
            </button>
            <div className="app-brand">
              <span style={{ width: 24, height: 24, borderRadius: '50%', background: 'var(--cm-coffee)', display: 'inline-block' }} />
              CoffeeMind GT · dashboard
            </div>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <span className={`chip ${backendStatus === 'online' ? 'chip--ok' : backendStatus === 'offline' ? 'chip--risk' : 'chip--neutral'}`}>
              backend {backendStatus}
            </span>
            <button className="btn btn--ghost" aria-label="Volver al inicio" onClick={onBackToLanding}>
              <IconHome /> Inicio
            </button>
          </div>
        </header>

        <aside className={`app-sidebar${sidebarOpen ? ' app-sidebar--open' : ''}`}>
          <div>
            <div className="sidebar__title">Parcelas ({parcelas.length})</div>
            {parcelas.map((p) => (
              <div key={p.id} className={`parcela-item${p.id === selectedId ? ' parcela-item--active' : ''}`}>
                <button
                  className={`parcela-pill ${p.id === selectedId ? 'parcela-pill--active' : ''}`}
                  onClick={() => { setSelectedId(p.id); setSidebarOpen(false) }}
                  style={{ width: '100%' }}
                >
                  <span className="parcela-pill__name">{p.nombre}</span>
                  <span className="parcela-pill__meta">
                    {p.region || 'sin región'} · {p.altitud} m · pH {p.ph} · {SHADE_LABELS[p.sombra]}
                  </span>
                </button>
                <div className="parcela-item__actions">
                  <button
                    className="btn btn--ghost btn--sm"
                    aria-label={`Editar ${p.nombre}`}
                    title="Editar"
                    onClick={() => { setEditing(p); setSidebarOpen(false) }}
                  >
                    <IconEdit />
                  </button>
                  <button
                    className="btn btn--ghost btn--sm"
                    aria-label={`Eliminar ${p.nombre}`}
                    title="Eliminar"
                    style={{ color: 'var(--cm-rust)' }}
                    onClick={() => setDelConfirm(p.id)}
                  >
                    <IconTrash />
                  </button>
                </div>
              </div>
            ))}
          </div>

          <button className="btn btn--primary" onClick={addParcela} style={{ width: '100%', justifyContent: 'center' }}>
            <IconPlus /> Agregar parcela
          </button>
        </aside>

        <main className="app-main">
          {selected && (
            <>
              <div className="app-main__header">
                <div>
                  <h2 style={{ marginBottom: '0.25rem' }}>{selected.nombre}</h2>
                  <div className="muted small">
                    {selected.region || 'sin región'} · altitud {selected.altitud} m ·
                    pH {selected.ph} · {SHADE_LABELS[selected.sombra]} ·
                    T {selected.temperature}°C · H {selected.humidity}% ·
                    lluvia {selected.rainfall} mm
                  </div>
                </div>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button className="btn btn--secondary btn--sm" aria-label={`Editar ${selected.nombre}`} onClick={() => setEditing(selected)}>
                    <IconEdit /> Editar
                  </button>
                  <button
                    className="btn btn--ghost btn--sm"
                    aria-label={`Eliminar ${selected.nombre}`}
                    style={{ color: 'var(--cm-rust)' }}
                    onClick={() => setDelConfirm(selected.id)}
                  >
                    <IconTrash /> Eliminar
                  </button>
                </div>
              </div>

              <div className="modules-grid">
                <CSPCard parcela={selected} />
                <BayesianCard parcela={selected} />
                <MLCard parcela={selected} />
                <MDPCard parcela={selected} />
              </div>
            </>
          )}
        </main>
      </div>

      {editing && (
        <ParcelaForm
          initial={editing}
          onSave={saveParcela}
          onCancel={() => setEditing(null)}
        />
      )}

      {delConfirm && (
        <div className="parcela-form-modal" onClick={() => setDelConfirm(null)}>
          <div className="confirm-panel" onClick={(e) => e.stopPropagation()}>
            <h3>¿Eliminar parcela?</h3>
            <p>
              Se eliminará <strong>{parcelas.find((p) => p.id === delConfirm)?.nombre}</strong>.{' '}
              Esta acción no se puede deshacer.
            </p>
            <div className="parcela-form__actions" style={{ marginTop: 'var(--space-6)' }}>
              <button className="btn btn--secondary" onClick={() => setDelConfirm(null)}>Cancelar</button>
              <button
                className="btn btn--primary"
                style={{ background: 'var(--cm-rust)', color: '#fff' }}
                onClick={() => { removeParcela(delConfirm); setDelConfirm(null) }}
              >
                <IconTrash /> Eliminar
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

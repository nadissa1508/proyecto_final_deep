import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { decide } from '../api'
import RustDecisionCard from './RustDecisionCard'

vi.mock('../api', () => ({ decide: vi.fn(), visionPredict: vi.fn() }))
afterEach(() => { cleanup(); vi.clearAllMocks() })

it('ignora una decisión tardía de la parcela anterior', async () => {
  let resolve
  decide.mockImplementation(() => new Promise((done) => { resolve = done }))
  const parcela = { id: 'a', altitud: 1500, sombra: 1 }
  const { rerender } = render(<RustDecisionCard parcela={parcela} regionKey="coban" />)
  fireEvent.click(screen.getByRole('button', { name: 'Decidir solo con el clima' }))
  rerender(<RustDecisionCard parcela={{ ...parcela, id: 'b' }} regionKey="coban" />)
  await act(async () => resolve({
    accion: 'nada', hojas: [], incidencia: null, creencia_solo_clima: { bajo: 1, medio: 0, alto: 0 },
    clima: { ventana_dias: 14, temperatura_media: 20, humedad_media: 85, lluvia_total_mm: 0,
      respecto_de_lo_habitual: {}, poco_confiable: false }, notas: [], advertencia: 'modelo',
  }))
  expect(screen.queryByText('Acción recomendada este mes')).toBeNull()
  expect(screen.getByRole('button', { name: 'Decidir solo con el clima' }).disabled).toBe(false)
})

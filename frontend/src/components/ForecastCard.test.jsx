import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { forecastGet } from '../api'
import ForecastCard from './ForecastCard'

vi.mock('../api', () => ({ forecastGet: vi.fn() }))
afterEach(() => { cleanup(); vi.clearAllMocks() })

const forecast = (asOf) => ({
  fuente_historia: 'archivo local', ultimo_dia_observado: asOf, dias_de_retraso: 0, dias: ['2025-09-19'],
  valores: { temperature_2m_mean: [20.8], relative_humidity_2m_mean: [92], precipitation_sum: [12.3] },
  poco_confiable: [false], notas: [],
})

it('sin fecha pronostica desde hoy', async () => {
  forecastGet.mockResolvedValue(forecast('2026-10-26'))
  render(<ForecastCard regionKey="atitlan" />)
  await waitFor(() => expect(forecastGet).toHaveBeenCalledWith('atitlan', 'live', undefined))
  expect(screen.getByText('Clima de los próximos 14 días')).toBeTruthy()
})

it('con fecha pide la consulta histórica y avisa al cambiarla', async () => {
  forecastGet.mockResolvedValue(forecast('2025-09-18'))
  const onAsOfChange = vi.fn()
  render(<ForecastCard regionKey="atitlan" asOf="2025-09-18" onAsOfChange={onAsOfChange} />)
  await waitFor(() => expect(forecastGet).toHaveBeenCalledWith('atitlan', 'live', '2025-09-18'))
  expect(screen.getByText('Clima de los 14 días desde el 2025-09-18')).toBeTruthy()
  expect(await screen.findByText('consulta histórica')).toBeTruthy()
  fireEvent.change(screen.getByLabelText('Fecha de consulta (opcional, histórica)'), { target: { value: '2025-06-01' } })
  expect(onAsOfChange).toHaveBeenCalledWith('2025-06-01')
})

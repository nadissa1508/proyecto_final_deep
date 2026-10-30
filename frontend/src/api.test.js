import { afterEach, describe, expect, it, vi } from 'vitest'
import { decide, forecastGet, visionPredict } from './api'

const okJson = (body) => ({ ok: true, json: async () => body })
const failed = (status, statusText, text) => ({ ok: false, status, statusText, text: async () => text })

afterEach(() => vi.unstubAllGlobals())

describe('api: subida de fotos', () => {
  it('decide envía multipart con región, fuente, campos opcionales y fotos', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({ accion: 'x' }))
    vi.stubGlobal('fetch', fetchMock)
    const foto = new File(['a'], 'h1.jpg', { type: 'image/jpeg' })
    await decide({ region: 'coban', files: [foto, foto], altitud: 1300, sombra: 2, humedadTipica: 80, source: 'local' })

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toMatch(/\/api\/decision$/)
    expect(init.method).toBe('POST')
    expect(init.headers).toBeUndefined() // el navegador pone el Content-Type con su separador
    const form = init.body
    expect(form.get('region')).toBe('coban')
    expect(form.get('source')).toBe('local')
    expect(form.get('altitud')).toBe('1300')
    expect(form.get('sombra')).toBe('2')
    expect(form.get('humedad_tipica')).toBe('80')
    expect(form.getAll('files')).toHaveLength(2)
  })

  it('decide omite los campos opcionales que no se dieron', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({}))
    vi.stubGlobal('fetch', fetchMock)
    await decide({ region: 'antigua' })
    const form = fetchMock.mock.calls[0][1].body
    expect(form.has('altitud')).toBe(false)
    expect(form.has('humedad_tipica')).toBe(false)
    expect(form.getAll('files')).toHaveLength(0)
  })

  it('visionPredict pide el mapa de calor solo si se indica', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({ hojas: [] }))
    vi.stubGlobal('fetch', fetchMock)
    const foto = new File(['a'], 'h.jpg', { type: 'image/jpeg' })
    await visionPredict([foto])
    await visionPredict([foto], { heatmap: true })
    expect(fetchMock.mock.calls[0][0]).toMatch(/heatmap=false/)
    expect(fetchMock.mock.calls[1][0]).toMatch(/heatmap=true/)
  })
})

describe('api: errores', () => {
  it('muestra el detalle que manda el backend (503 por pesos faltantes)', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(failed(503, 'Service Unavailable', JSON.stringify({ detail: 'Módulo de visión no disponible: faltan pesos' }))))
    await expect(visionPredict([])).rejects.toThrow(/503.*faltan pesos/)
  })

  it('serializa un detalle que no es texto', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(failed(422, 'Unprocessable', JSON.stringify({ detail: [{ msg: 'campo requerido' }] }))))
    await expect(forecastGet('coban')).rejects.toThrow(/422.*campo requerido/)
  })

  it('usa el texto plano si la respuesta de error no es JSON', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(failed(502, 'Bad Gateway', 'upstream caído')))
    await expect(forecastGet('coban')).rejects.toThrow(/502.*upstream caído/)
  })

  it('explica que no hay conexión con el backend', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    await expect(forecastGet('coban')).rejects.toThrow(/No se pudo conectar al backend/)
  })
})

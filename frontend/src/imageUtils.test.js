import { afterEach, describe, expect, it, vi } from 'vitest'
import { MAX_SIDE, prepareImage } from './imageUtils'

const jpeg = (name = 'hoja.jpg') => new File(['x'], name, { type: 'image/jpeg' })

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('prepareImage: formatos', () => {
  it('rechaza HEIC por tipo con una pista para el iPhone', async () => {
    const file = new File(['x'], 'IMG_1.jpg', { type: 'image/heic' })
    await expect(prepareImage(file)).rejects.toThrow(/HEIC.*Más compatible/s)
  })

  it('rechaza HEIC aunque el navegador no ponga el tipo (por extensión)', async () => {
    const file = new File(['x'], 'IMG_2.HEIC', { type: '' })
    await expect(prepareImage(file)).rejects.toThrow(/HEIC/)
  })

  it('rechaza formatos que el backend no acepta', async () => {
    const file = new File(['x'], 'doc.pdf', { type: 'application/pdf' })
    await expect(prepareImage(file)).rejects.toThrow(/JPEG, PNG o WebP/)
  })

  it('avisa si la imagen no se puede leer', async () => {
    vi.stubGlobal('createImageBitmap', vi.fn().mockRejectedValue(new Error('corrupta')))
    await expect(prepareImage(jpeg('rota.jpg'))).rejects.toThrow(/rota\.jpg: no se pudo leer/)
  })
})

describe('prepareImage: reducción', () => {
  function stubBrowser(width, height) {
    const bitmap = { width, height, close: vi.fn() }
    vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue(bitmap))
    const draw = vi.fn()
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ drawImage: draw })
    vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((cb) => cb(new Blob(['jpg'], { type: 'image/jpeg' })))
    URL.createObjectURL = vi.fn(() => 'blob:prueba')
    return { bitmap, draw }
  }

  it('reduce una foto grande al lado máximo y conserva la proporción', async () => {
    const { bitmap, draw } = stubBrowser(4128, 2322)
    const out = await prepareImage(jpeg('grande.jpg'))
    const [, , , w, h] = draw.mock.calls[0]
    expect(Math.max(w, h)).toBe(MAX_SIDE)
    expect(w / h).toBeCloseTo(4128 / 2322, 2)
    expect(out.file.type).toBe('image/jpeg')
    expect(out.file.name).toBe('grande.jpg')
    expect(out.url).toBe('blob:prueba')
    expect(bitmap.close).toHaveBeenCalled()
  })

  it('no agranda una foto pequeña', async () => {
    const { draw } = stubBrowser(640, 480)
    await prepareImage(jpeg('chica.jpg'))
    const [, , , w, h] = draw.mock.calls[0]
    expect([w, h]).toEqual([640, 480])
  })

  it('cambia la extensión a .jpg al convertir un PNG', async () => {
    stubBrowser(100, 100)
    const png = new File(['x'], 'captura.png', { type: 'image/png' })
    const out = await prepareImage(png)
    expect(out.file.name).toBe('captura.jpg')
  })
})

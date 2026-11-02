import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import LeafUploader, { MIN_RECOMMENDED } from './LeafUploader'

afterEach(() => cleanup())

const items = (n) => Array.from({ length: n }, (_, i) => ({ id: `id${i}`, file: new File(['x'], `h${i}.jpg`), url: `blob:${i}` }))

describe('LeafUploader', () => {
  it('ofrece elegir archivos (sin botón de cámara)', () => {
    const { container } = render(<LeafUploader items={[]} onChange={() => {}} />)
    expect(screen.getByText('Elegir archivos')).toBeTruthy()
    expect(screen.queryByText('Tomar foto')).toBeNull()
    expect(container.querySelector('input[capture]')).toBeNull()
    const picker = container.querySelector('input[type=file]')
    expect(picker.getAttribute('accept')).toBe('image/jpeg,image/png,image/webp')
    expect(picker.multiple).toBe(true)
  })

  it('avisa que se recomiendan al menos %d hojas cuando hay menos' .replace('%d', MIN_RECOMMENDED), () => {
    render(<LeafUploader items={items(3)} onChange={() => {}} />)
    expect(screen.getByText('3 hojas').className).toMatch(/chip--warn/)
    expect(screen.getByText(`Se recomiendan al menos ${MIN_RECOMMENDED}.`)).toBeTruthy()
  })

  it('no muestra el aviso con las hojas recomendadas', () => {
    render(<LeafUploader items={items(MIN_RECOMMENDED)} onChange={() => {}} />)
    expect(screen.getByText(`${MIN_RECOMMENDED} hojas`).className).toMatch(/chip--ok/)
    expect(screen.queryByText(/Se recomiendan al menos/)).toBeNull()
  })

  it('rechaza un archivo HEIC con el mensaje para el usuario y no agrega nada nuevo', async () => {
    const onChange = vi.fn()
    const { container } = render(<LeafUploader items={[]} onChange={onChange} />)
    const picker = container.querySelector('input[type=file]')
    const heic = new File(['x'], 'IMG_9.heic', { type: 'image/heic' })
    fireEvent.change(picker, { target: { files: [heic] } })
    await waitFor(() => expect(screen.getByText(/formato HEIC no soportado/)).toBeTruthy())
    expect(onChange).toHaveBeenCalledWith([])
  })

  it('quitar una hoja la elimina de la lista', () => {
    URL.revokeObjectURL = vi.fn()
    const onChange = vi.fn()
    render(<LeafUploader items={items(2)} onChange={onChange} />)
    fireEvent.click(screen.getByLabelText('Quitar hoja 1'))
    expect(onChange).toHaveBeenCalledWith([expect.objectContaining({ id: 'id1' })])
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:0')
  })

  it('con el componente deshabilitado no deja quitar ni vaciar', () => {
    render(<LeafUploader items={items(2)} onChange={() => {}} disabled />)
    expect(screen.getByLabelText('Quitar hoja 1').disabled).toBe(true)
    expect(screen.getByText('Quitar todas').disabled).toBe(true)
  })
})

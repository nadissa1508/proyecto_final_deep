// Preparación de fotos en el navegador antes de subirlas: reducir a ~1024 px (una foto de celular pesa varios MB),
// corregir la orientación EXIF y avisar de formatos que el backend no acepta (HEIC).

export const MAX_SIDE = 1024
export const MAX_FILES = 40
const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp']

function isHeic(file) {
  return /heic|heif/i.test(file.type) || /\.(heic|heif)$/i.test(file.name)
}

/** Devuelve { file, url } con un JPEG reducido, o lanza un Error con un mensaje para el usuario. */
export async function prepareImage(file) {
  if (isHeic(file)) {
    throw new Error(`${file.name}: formato HEIC no soportado. En el iPhone elija "Más compatible" en Ajustes > Cámara > Formatos, o convierta la foto a JPEG.`)
  }
  if (!ACCEPTED.includes(file.type)) {
    throw new Error(`${file.name}: use una foto JPEG, PNG o WebP.`)
  }
  let bitmap
  try {
    // 'from-image' aplica la orientación EXIF (las fotos del celular suelen venir rotadas en los datos)
    bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' })
  } catch {
    throw new Error(`${file.name}: no se pudo leer la imagen.`)
  }
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height))
  const canvas = document.createElement('canvas')
  canvas.width = Math.round(bitmap.width * scale)
  canvas.height = Math.round(bitmap.height * scale)
  canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height)
  bitmap.close?.()
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.9))
  if (!blob) throw new Error(`${file.name}: no se pudo reducir la imagen.`)
  const name = file.name.replace(/\.[^.]+$/, '') + '.jpg'
  const out = new File([blob], name, { type: 'image/jpeg' })
  return { file: out, url: URL.createObjectURL(out) }
}

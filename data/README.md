# Datos

Todo lo que no está en `raw/*.csv` se descarga con scripts y **no se versiona** (ver `.gitignore`). Para reproducir desde cero:

```bash
pip install -r requirements.txt -r requirements-dl.txt
```

```bash
python -m src.vision.download
```

```bash
python -m src.vision.datasets
```

```bash
python -m src.forecast.download
```

(En PowerShell, definir antes `$env:PYTHONPATH = "."`.)

## raw/ — heredados del proyecto de IA (versionados)

| Archivo | Origen | Uso |
|---|---|---|
| `crop_recommendation.csv` | Kaggle | Módulo ML y CPTs de la red bayesiana |
| `coffee_quality.csv` | Coffee Quality Institute | Módulo CSP |
| `varieties.csv` | Elaborado por el equipo | Dominio del CSP |

## raw/rocole/ — RoCoLe (robusta, Ecuador)

- **Fuente:** Mendeley Data, DOI [10.17632/c5yvn32dzg.2](https://data.mendeley.com/datasets/c5yvn32dzg/2), CC BY 4.0. Parraga-Alava et al. (2019), *Data in Brief*.
- **Descarga:** `src/vision/download.py`, API pública de Mendeley, cada archivo verificado con SHA-256. Se omite `RoCoLe-voc.tar.gz` (~700 MB, anotaciones VOC redundantes).
- **Contenido verificado:** 1,560 JPG (4128×2322 o 2048×1152, fotos de campo con celular) + `RoCoLe-classes.xlsx` (etiquetas) + anotaciones de segmentación (CSV/JSON/COCO).
- **Etiquetas (coinciden con el artículo):** healthy 791 · rust_level_1 344 · rust_level_2 166 · rust_level_3 62 · rust_level_4 30 · red_spider_mite 167.
- **Planta:** el nombre `C<c>P<p><E|H><n>.jpg` identifica la planta (`C<c>P<p>`) y la cara de la hoja (E = envés, H = haz). 390 plantas × 4 fotos. Un archivo mal nombrado: `C11P11HE.jpg` (es de la planta C11P11).
- **Problema de calidad encontrado:** 67 fotos aparecen dos veces bajo nombres de plantas distintas (copia byte a byte), y en 21 de ellas con clases distintas (p. ej. la misma foto como `healthy` y como `rust_level_1`). Ejemplos en `docs/figures/rocole_duplicados_contradictorios.png`. Tratamiento en `src/vision/datasets.py`.

## raw/bracol/ — BRACOL (arábica, Brasil)

- **Fuente:** Esgario, Krohling y Ventura (2020), *Computers and Electronics in Agriculture* (arXiv:1907.11561). Registro en Mendeley, DOI [10.17632/yy2k5y8mxg.1](https://data.mendeley.com/datasets/yy2k5y8mxg/1).
- **Problema encontrado:** el zip de Mendeley (165 MB) está **truncado en origen**: su SHA-256 coincide con el que publica Mendeley, pero no tiene directorio central y solo se recuperan 1,402 de 1,747 hojas, sin el dataset de síntomas.
- **Descarga usada:** la copia completa que enlazan los autores en [github.com/esgario/lara2018](https://github.com/esgario/lara2018) (Google Drive, `coffee-datasets.zip`, 279 MB, SHA-256 `e6fc32d5…c46c9f89`). Se descarga con `gdown` y se descomprime en `extracted/coffee-datasets/`.
- **Contenido verificado:** `leaf/` (1,747 hojas 2048×1024 por el envés sobre fondo blanco, `dataset.csv`, `legend.txt`), `symptom/` (recortes de síntomas con train/val/test) y `segmentation/` (500 imágenes con máscaras).
- **Etiquetas (`leaf/dataset.csv`):** `predominant_stress` (0 sana, 1 minador, 2 roya, 3 phoma, 4 cercospora, 5 indeterminado), booleanos de presencia por estrés y `severity` 0–4 del estrés predominante. Hojas con roya predominante por severidad: 1 → 313, 2 → 134, 3 → 45, 4 → 39. 372 hojas tienen más de un estrés.

## processed/vision/ — manifiesto de imágenes

`python -m src.vision.datasets` genera `manifest.csv`: una fila por imagen con ruta, dataset, grupo para el split, etiqueta original, severidad, clase del proyecto (0 sana, 1 roya leve, 2 roya severa, 3 otra afección) y motivo de exclusión. Las reglas están documentadas en el docstring de `src/vision/datasets.py`.

| Clase | BRACOL | RoCoLe | Total |
|---|---|---|---|
| 0 sana | 272 | 757 | 1,029 |
| 1 roya leve (≤ 5 %) | 312 | 324 | 636 |
| 2 roya severa (> 5 %) | 218 | 231 | 449 |
| 3 otra afección | 783 | 155 | 938 |
| **Total usable** | **1,585** | **1,467** | **3,052** |

Excluidas (255): BRACOL estrés indeterminado 62, BRACOL roya presente pero no predominante 99, BRACOL copia exacta 1, RoCoLe copias exactas redundantes 46, RoCoLe duplicados con etiquetas contradictorias 47.

Detección de duplicados: SHA-256 en ambos datasets; pHash (Hamming ≤ 4) solo en RoCoLe, porque en BRACOL marcaba como iguales hojas distintas por el fondo blanco (ver `docs/figures/bracol_falsos_casi_duplicados_phash.png`).

## raw/openmeteo/ — clima (Open-Meteo)

- **Fuente:** [Open-Meteo](https://open-meteo.com/), datos CC BY 4.0, uso no comercial gratuito.
- **Regiones:** `configs/regions.yaml`: una **estación de ANACAFE por región** (lat, lon y altitud copiadas del portal). El script se niega a correr si falta alguno de esos tres datos.
- **Altitud (verificado):** Open-Meteo corrige la temperatura a la altitud que se le indique (`elevation`, por defecto el relieve de 90 m del punto). Huehuetenango, 2024, mismo punto: 21.8 °C a 998 m (por defecto), 19.6 °C con la altura media de la celda (1,631 m) y **17.8 °C con `elevation=1900`**, que coincide con la estación de Meteostat a 1,901 m (17.6 °C). El error de la primera descarga no fue de la grilla: el punto de La Democracia está a 998 m y el café crece a 1,500–2,000 m. Por eso la altitud real de cada finca es un dato obligatorio.
- **Histórico:** `era5_seamless` diario desde 1980-01-01 hasta hoy − 7 días: temperatura media/máx./mín., humedad relativa, punto de rocío, precipitación, horas de lluvia y evapotranspiración. Archivo `era5_<region>.csv`. Combina ERA5-Land (~11 km) para temperatura y humedad con ERA5 (~25 km) para lluvia y evapotranspiración.
- **Corrección:** el comparar con la estación de Meteostat de Huehuetenango (1,901 m, 17.6 °C) no era válido: esa estación está en la ciudad, y la de ANACAFE (Rancho Carmela, 966 m) mide 22.4 °C. Con las coordenadas y la altitud de la estación ANACAFE, el sesgo de temperatura en esa región es −0.6 °C.
- **Problema encontrado:** `models=era5_land` devuelve `null` en **todos** los días de `precipitation_sum` y `et0_fao_evapotranspiration` para Guatemala (y `precipitation_hours = 0`). El primer archivo descargado así parecía válido, pero la lluvia estaba vacía. Por eso se usa `era5_seamless` y el script valida que no haya fechas faltantes, más de 1 % de nulos por variable, ni lluvia anual fuera de 200–6,000 mm.
- **Pronósticos emitidos:** Previous Runs API desde 2024-01-01, lo que el modelo numérico pronosticó 1–7 días antes, agregado a diario. Archivo `previous_runs_<region>.csv`.
- **Cupo:** la descarga completa pesa ~8,250 llamadas ponderadas (cupo diario 10,000). El script se autolimita a ~4,000/hora y guarda cada bloque en `_cache/`: si se corta, se vuelve a correr y continúa.

## raw/power/ — NASA POWER (contraste)

- **Fuente:** [NASA POWER](https://power.larc.nasa.gov/) diario desde 1981 (MERRA-2 y satélite; grilla ~0.5° x 0.625°). Consulta por coordenadas, sin API key. Probado para Antigua 2020: 366 días sin nulos.
- **Descarga:** `python -m src.forecast.download_power` (cuando `regions.yaml` esté completo). Archivos `power_<region>.csv`.
- **Limitación:** no recibe altitud; su temperatura corresponde a la altura media de su celda. Sirve para contrastar variaciones y lluvia con Open-Meteo, no para el nivel absoluto de temperatura.
- **Descarga hecha:** 6 regiones, 16,734 días (1981 a oct-2026). La radiación solar no existe de 1981 a 1983 (el sensor de satélite empieza en 1984); es la única excepción que acepta la validación.
- **Comparación contra las estaciones de ANACAFE (2022–2025, `python -m src.forecast.compare_sources`, resultado en `experiments/forecast/source_comparison.csv`):**
  - *Temperatura media:* ERA5 con la altitud de la estación tiene sesgo de −0.6 a +2.2 °C y correlación diaria de 0.74–0.89; POWER sesga +3 a +4 °C (−4.3 en Huehuetenango) por no corregir altitud.
  - *Humedad:* ERA5 correlaciona mejor en las 6 regiones (0.65–0.86 frente a 0.46–0.79); las dos subestiman la humedad medida (2 a 10 puntos).
  - *Lluvia:* ERA5 correlaciona mejor en 5 de 6 regiones, a nivel diario (0.38–0.55) y mensual (0.77–0.93). Ninguna fuente acierta los totales: el cociente fuente/estación va de 0.56 a 1.63 en ERA5 y de 0.56 a 1.48 en POWER.
  - *Conclusión:* ERA5 es la fuente de entrenamiento; POWER solo sirve de contraste. La lluvia es la variable débil, lo que justifica afinar con las estaciones.

## Baselines del pronóstico (`python -m src.forecast.baselines` → `experiments/forecast/baselines.csv`)

Split temporal (`configs/forecast.yaml`): train hasta 2017, val 2018–2020, test desde 2021. Ventana de 90 días, salida de 14 días, normalización solo con train, `log1p` a la lluvia. MAE contra ERA5 en el test:

- **Temperatura:** la persistencia gana a todos los horizontes en 4 de 6 regiones (0.4–0.65 °C a 1 día; 0.8–1.4 °C a 14 días); el estacional ingenuo (0.8–1.2 °C) gana a 14 días en Cobán, Huehuetenango y Nuevo Oriente.
- **Humedad:** persistencia mejor a 1–3 días, estacional mejor desde ~7 días.
- **Lluvia:** el estacional (climatología) iguala o supera a la persistencia en casi todas las regiones; la lluvia de ayer predice poco la de mañana. Nota: para el error absoluto el baseline estacional correcto usa la **mediana**, no la media (4.23 frente a 4.63 mm/día); `experiments/forecast/baselines.csv` solo trae la versión con media, y los 4 baselines completos están en `docs/resultados_forecast.md`.
- **Pronóstico numérico de Open-Meteo (solo 1–7 días, desde 2024):** **no es un techo limpio.** Medido contra ERA5 pierde frente a la persistencia en temperatura (1.11 °C frente a 0.53 °C a 1 día). Parte es sesgo (Huehuetenango +1.7 °C), pero en Antigua y Atitlán el error se mantiene sin sesgo (≈1.1–1.7 °C): son dos modelos distintos con topografía distinta. La comparación justa es contra las estaciones de ANACAFE; está pendiente.

## raw/anacafe/ — estaciones de ANACAFE (descarga manual, no versionada)

- **Fuente:** portal [meteorologia.anacafe.org](https://meteorologia.anacafe.org/) (ClimaCafé), 173 estaciones en fincas cafetaleras de 7 regiones. Requiere iniciar sesión y llenar un formulario; los datos se exportan a CSV por estación. **No se suben al repositorio** (público): cada integrante debe pedirlos con su propia cuenta y aceptar los términos del portal.
- **Archivos esperados:** `data/raw/anacafe/estacion_<cod>.csv`, con `<cod>` el código de la estación en el portal (columna Cod).
- **Formatos que ofrece el portal:** resumen mensual (una fila por mes) y **registros diarios** con lluvia (mm), humedad (%), viento (kph), temperatura media/máx./mín./de rocío (°C), radiación máxima (W/m²) y horas de sol, por estación y año. Hay datos desde 2018.
- **Estaciones del pronóstico (CSV diarios de 2026 recibidos):** 15 Rancho Carmela (966 m), 3 Bella Vista (1,521 m), 163 Oficina Regional VI, Cobán (1,332 m), 71 Cooperativa Nahualá (1,197 m), 8 La Mandarina, Fraijanes (1,713 m) y 130 Chiramay, Chiquimula (1,399 m). Altitudes: relieve de 90 m en las coordenadas de la estación.
- **Archivos:** `*DIARIO*.csv` (diario, uno por estación y año; columnas `Dia, ID_ESTACION, NOMBRE, LONGITUD, LATITUD, Lluvia_mm, HumedadMedia, VtokphMedia, TempMedia, TempMax, TempMin, TempRocioMedia, RadSolarMax, HorasSol`). Los `MENSUAL` se ignoran. Para otro año, cambiar `y=` en la URL del portal (p. ej. `resumen_mes_dt.html?id=15&y=2025`; la vista diaria usa el mismo patrón) y exportar. `src/forecast/anacafe.py` lee todos los CSV diarios sin importar el nombre.
- **Validación de Open-Meteo contra estas estaciones (ene–oct 2026, `era5_seamless` con la altitud de la estación):** sesgo de temperatura entre −0.6 y +1.7 °C (correlación diaria 0.76–0.97); humedad entre 0.6 y 12 puntos por debajo de lo medido; **lluvia es lo débil**: correlación diaria 0.19–0.61 y totales entre −39 % y +70 % respecto al pluviómetro. Con eso, el afinado con datos de estación (transfer learning) tiene justificación.
- **Cobertura recibida (2020 a oct-2026, 12,956 días en total):** 3 Bella Vista, 15 Rancho Carmela, 71 Nahualá y 130 Chiramay desde enero/marzo de 2020; 8 La Mandarina desde marzo de 2020; **163 Cobán desde el 16-dic-2021**. Periodo sólido para las 6: **2022–2026**. Huecos grandes: La Mandarina sin ninguna fila de feb-2023 a ene-2024; Rancho Carmela 2020 casi vacío (feb, mar y may sin datos); Nahualá 2020 con 34 % de días; Bella Vista 2021 con 50 % de lluvia válida. Viento de Nahualá muerto (2 % válido en 2026).
- **Lecturas imposibles reales detectadas:** temperatura media de 67–72 °C (Bella Vista, oct-2021), máxima de 137 °C, mínima de −32 °C, punto de rocío de −110 °C (Rancho Carmela), 31 días con la misma temperatura (Rancho Carmela, dic-2020).
- **Control de calidad necesario (visto en muestras):** Sabanetas (cod 135) marca 0 mm de lluvia de julio a octubre de 2026 (temporada de lluvias): pluviómetro dañado o sin lectura. Coatepeque (cod 1) tiene viento exactamente 0 desde el 22-ene-2026: anemómetro sin lectura. Hay que filtrar rachas de ceros y valores constantes antes de comparar o entrenar.
- **Uso previsto:** (1) medir el sesgo de Open-Meteo contra datos medidos en finca; (2) afinar el pronóstico con estas series tras preentrenarlo con ERA5 (transfer learning, S9).

## raw/insivumeh/ — pendiente

Solicitud de datos de estaciones por el portal de trámites de INSIVUMEH. No bloquea nada; si llega, servirá para medir el sesgo de ERA5 en 1–2 estaciones.

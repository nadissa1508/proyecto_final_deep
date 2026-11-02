# Cómo reproducir el proyecto

Orden recomendado. Los datos, las imágenes y los pesos **no** están en el repositorio: cada paso los regenera. Los tiempos son los medidos en
una laptop con una RTX 3050 de 4 GB (con freno térmico) y son aproximados.

## 0. Entorno

```bash
python -m venv .venv && . .venv/Scripts/activate        # Windows (Git Bash); en PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dl.txt
# Para entrenar con GPU, instale PyTorch con CUDA en vez del de requirements-dl.txt:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

En PowerShell defina antes `$env:PYTHONPATH = "."`. Pruebas: `pytest tests/` (81; no necesitan pesos ni internet) y, en `frontend/`, `npm test` (24, vitest).

## 1. Datos

```bash
python -m src.vision.download                  # RoCoLe (Mendeley) y BRACOL (copia de los autores en Google Drive); verifica SHA-256
python -m src.forecast.download                # ERA5 (Open-Meteo) de las 6 regiones y pronósticos previos; respeta el cupo, es reanudable
python -m src.forecast.download_power          # NASA POWER, solo como contraste
```

Las estaciones de ANACAFE **no** tienen descarga automática: exporte los CSV diarios del portal ClimaCafé a `data/raw/anacafe/` (uno por
estación y año; ver el docstring de `src/forecast/anacafe.py`). `python -m src.forecast.anacafe` imprime el control de calidad y
`python -m src.forecast.compare_sources` compara ERA5, POWER y las estaciones. Las coordenadas y altitudes de las regiones están en
`configs/regions.yaml`. Detalle de cada fuente en `data/README.md`.

> Varios módulos **no** usan `argparse` y ejecutan su trabajo al llamarlos (por ejemplo `src.vision.datasets`, `splits`, `summarize`,
> `finalize`, `src.forecast.baselines`, `evaluate`, `station_eval`, `download_power`): no los llame con `--help`.

## 2. Visión (CNN de nivel de roya)

```bash
python -m src.vision.datasets                  # manifiesto y mapeo de etiquetas (un solo lugar), duplicados
python -m src.vision.splits                    # split sin fuga y baselines
python -m src.vision.preprocess                # caché a 320 px
python -m src.vision.train --config configs/vision_resnet18_dlr_both.yaml            # una corrida; semilla 42
python -m src.vision.train --config configs/vision_resnet18_dlr_both.yaml --seed 1   # repetir con --seed 1, 2, 3
python -m src.vision.train --config configs/vision_efficientnet_b0_dlr_both.yaml --seed 1
python -m src.vision.summarize                 # media ± desviación entre semillas
python -m src.vision.finalize                  # calibración, selección por validación y exportación de models/vision/artifact.json
python -m src.vision.validate_explanations     # Grad-CAM: aleatorización, deletion/insertion, fondo (reanudable; usa la GPU)
python -m src.vision.check_infer               # infer.py reproduce la evaluación
python -m src.vision.abstention --write        # regla de «foto no concluyente» (repetir tras cada finalize)
```

Otras configuraciones de la ablación: `configs/vision_*.yaml`. Una corrida de ResNet-18 tarda de 2 a 10 minutos según el freno térmico.
Resultados en `docs/resultados_vision.md` y `notebooks/06_eda_vision.ipynb`, `07_vision.ipynb`.

## 3. Pronóstico (LSTM, GRU, Transformer)

```bash
python -m src.forecast.baselines               # persistencia, estacional (media/mediana) y estacional + anomalía
python -m src.forecast.train --config configs/forecast_gru.yaml              # y _lstm, _transformer; --seed 1, 2, 3 (≈ 1 min cada una)
python -m src.forecast.evaluate                # resumen entre semillas contra los baselines
python -m src.forecast.station_eval            # contra las estaciones de ANACAFE
python -m src.forecast.finetune                # afinado con las estaciones (cabeza o completo)
python -m src.forecast.loso                    # validación en una estación no vista; --cut corte2 para el otro corte de fechas
python -m src.forecast.latency                 # costo del retraso de ERA5 en operación
python -m src.forecast.export                  # models/forecast/artifact.json
python -m src.forecast.check_infer
```

Resultados en `docs/resultados_forecast.md` y `notebooks/08_eda_clima.ipynb`. **Un solo trabajo de GPU a la vez** (con dos a la vez la GPU de
4 GB casi no avanza).

## 4. Integración

```bash
python -m src.fusion.build_evidence            # terciles y error de la ventana con el pronosticador real (≈ 25 min)
python -m src.fusion.simulate --reps 300 --stale 0 5
python -m src.fusion.simulate --reps 120 --stale 0 --sensitivity
python -m src.fusion.simulate --reps 150 --stale 0 --slopes 0.1 0.25 0.5 --gammas 0.0 0.15 0.3 0.6
python -m src.fusion.sensitivity
python -m src.fusion.diagnose_climate
python -m src.fusion.evaluate_favorability     # alarma preventiva: el pronóstico frente al evento «días favorables»
```

Supuestos en `configs/fusion.yaml`; resultados en `docs/resultados_fusion.md` y `notebooks/10_integracion.ipynb`.

## 5. Aplicación

```bash
uvicorn src.api.main:app --reload --port 8000  # carga visión, pronóstico y fusión; si faltan pesos, sus endpoints responden 503
cd frontend && npm install && npm run dev      # http://localhost:5173
```

Docker (PyTorch solo CPU; los pesos y `models/fusion/` se montan como volumen, no van en la imagen):

```bash
docker compose up --build                      # backend en :8000, frontend en :5173
```


## 6. Pesos, figuras y revisión antes de publicar

```bash
python -m src.common.weights pack              # dist/coffeemind_weights.zip (solo lo que usan los artefactos) y models/weights_manifest.json
python -m src.common.weights download --url <URL del zip del release>   # en un clon limpio; o --zip <archivo local>
python -m src.common.weights verify            # compara con el manifiesto (SHA-256)
python -m src.common.figures                   # regenera las figuras del informe en docs/figures/
python -m src.common.publish_check             # archivos que no deben subirse, secretos, rutas locales, archivos obligatorios
```

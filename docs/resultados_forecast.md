# Resultados del pronóstico climático (Fase 3, primera ablación)

Fuente de verdad: `experiments/forecast/summary.csv`, `wins.csv` y `<corrida>/test_mae.csv` (se regeneran con `python -m src.forecast.evaluate`).
Configuración: `configs/forecast.yaml` y `configs/forecast_{lstm,gru,transformer}.yaml`.

## Diseño

- Un modelo global por arquitectura, entrenado con las ventanas de las 6 regiones juntas (cada región normalizada con sus estadísticas de train). Entrada: 90 días de ERA5 (8 variables + sin/cos del día del año); salida: los 14 días siguientes de temperatura media, humedad relativa y lluvia, de una sola vez.
- Split temporal: train hasta 2017, validación 2018–2020, **test desde 2021** (ERA5, no estaciones). Lluvia con `log1p` en el entrenamiento; el error se mide en °C, puntos de humedad y mm/día.
- 4 corridas por arquitectura (semillas 42, 1, 2, 3). LSTM y GRU de 2 capas × 128 unidades; Transformer de 3 capas, 4 cabezas, d=64. AdamW, Huber, cosine, early stopping (paciencia 6).
- Para cada horizonte solo cuentan los días pronosticados dentro del periodo de test, y todos los modelos y baselines se evalúan **en exactamente las mismas muestras**.

## Baselines (el rival a vencer es el mejor de los cuatro)

| Baseline | Qué hace |
|---|---|
| Persistencia | repite el valor de hoy |
| Estacional (media) | climatología del día del año, 1980–2017, ±7 días |
| Estacional (mediana) | lo mismo con la mediana: el predictor óptimo para el error absoluto |
| Estacional + anomalía | climatología más la anomalía media de los últimos 30 días |

**Por qué cuatro.** Con solo persistencia y estacional (media) los modelos parecían ganar por mucho (temperatura a 14 días: −17 %). Dos defectos de esos baselines lo inflaban:
1. **La climatología de 1980–2017 está entre 0.4 y 1.1 °C más fría que el periodo de test** (el clima se calentó). Eso arruina su error en temperatura, y es lo que corrige la anomalía reciente.
2. **La media no es el predictor óptimo del error absoluto en la lluvia** (muchos días secos y una cola larga). Usar la mediana baja el MAE del baseline de 4.63 a 4.23 mm/día.

## Resultados: MAE medio de las 6 regiones, media ± desviación entre 4 semillas

Entre paréntesis, mejora respecto al mejor baseline (negativo = pierde).

| Variable | h | Mejor baseline | GRU | LSTM | Transformer |
|---|---|---|---|---|---|
| Temperatura (°C) | 1 | 0.52 (persistencia) | 0.52 ± 0.01 (+1 %) | 0.53 ± 0.01 (0 %) | 0.53 ± 0.01 (0 %) |
| | 3 | 0.83 (est.+anom.) | 0.77 ± 0.01 (+7 %) | 0.78 ± 0.02 (+6 %) | 0.76 ± 0.02 (+8 %) |
| | 7 | 0.87 (est.+anom.) | 0.84 ± 0.01 (+3 %) | 0.86 ± 0.03 (+1 %) | 0.83 ± 0.02 (+4 %) |
| | 14 | 0.90 (est.+anom.) | 0.89 ± 0.01 (+2 %) | 0.89 ± 0.02 (+2 %) | 0.87 ± 0.02 (+3 %) |
| Humedad (pts) | 1 | 3.78 (persistencia) | 3.61 ± 0.02 (+4 %) | 3.67 ± 0.04 (+3 %) | 3.63 ± 0.01 (+4 %) |
| | 3 | 5.18 (est.+anom.) | 4.81 ± 0.01 (+7 %) | 4.88 ± 0.06 (+6 %) | 4.85 ± 0.03 (+6 %) |
| | 7 | 5.30 (estacional) | 5.14 ± 0.03 (+3 %) | 5.23 ± 0.10 (+1 %) | 5.16 ± 0.03 (+3 %) |
| | 14 | 5.30 (estacional) | 5.34 ± 0.02 (−1 %) | 5.44 ± 0.09 (−3 %) | 5.35 ± 0.05 (−1 %) |
| Lluvia (mm/día) | 1 | 4.23 (mediana) | 3.78 ± 0.02 (+10 %) | 3.80 ± 0.02 (+10 %) | 3.75 ± 0.01 (+11 %) |
| | 3 | 4.23 (mediana) | 4.24 ± 0.01 (0 %) | 4.27 ± 0.05 (−1 %) | 4.21 ± 0.02 (+1 %) |
| | 7 | 4.24 (mediana) | 4.31 ± 0.02 (−2 %) | 4.39 ± 0.08 (−4 %) | 4.30 ± 0.04 (−1 %) |
| | 14 | 4.25 (mediana) | 4.37 ± 0.03 (−3 %) | 4.42 ± 0.10 (−4 %) | 4.36 ± 0.04 (−3 %) |

Victorias sobre el mejor baseline (de 72 combinaciones región × variable × horizonte, media entre 4 corridas): GRU 47.2, Transformer 45.8, LSTM 42.0.

## Qué se puede afirmar

1. **El aporte es modesto y concentrado en el corto plazo.** Las mejoras reales son 6–8 % en temperatura y humedad a 3 días y ~10 % en lluvia a 1 día. A 14 días solo se mejora la temperatura (2–4 %), y en humedad y lluvia el modelo **no supera** a la climatología.
2. **A 1 día en temperatura no hay mejora** frente a la persistencia (0.52 °C).
3. **GRU y Transformer están empatados** (diferencias de 0.01–0.02 °C, dentro de la desviación entre semillas). LSTM es algo peor y más variable. No hay evidencia de que el Transformer, más moderno, gane en series de este tamaño.
4. Las desviaciones entre semillas (0.01–0.10) son pequeñas frente a las diferencias con los baselines a 1–3 días, pero **comparables a ellas a 7–14 días**, donde las ganancias de 1–4 % no son concluyentes.

## Barrido de hiperparámetros contra el sobreajuste (GRU, semilla 42, elegido solo por validación)

Las curvas confirman sobreajuste: la pérdida de entrenamiento baja en cada época (0.245 → 0.184) y la de validación sube desde la época 2 (0.225 → 0.278). Se probaron seis configuraciones (`experiments/forecast/sweep_gru/`):

| Configuración | Mejor época | Pérdida de val | Pérdida de val final |
|---|---|---|---|
| lr 3e-4 | 8 | 0.2249 | 0.2494 |
| lr 3e-4 + dropout 0.3 | 8 | 0.2261 | 0.2429 |
| 10 épocas, lr 5e-4, dropout 0.3 | 4 | 0.2261 | 0.2313 |
| dropout 0.3 | 2 | 0.2267 | 0.2684 |
| **base** (lr 1e-3, dropout 0.1) | 2 | 0.2293 | 0.2687 |
| lr 3e-4 + dropout 0.3 + ruido de entrada 0.1 | 8 | 0.2279 | 0.2417 |
| hidden 64, dropout 0.3, weight decay 0.05 | 3 | 0.2326 | 0.2368 |

**Conclusión:** bajar el learning rate retrasa el sobreajuste (la mejor época pasa de 2 a 8) pero **el piso de la pérdida de validación casi no se mueve** (0.225–0.233, una mejora de ~2 %, comparable a la variación entre semillas del modelo base: 0.225–0.231). El techo lo marca la información disponible en una ventana de ERA5, no la regularización. Por eso no se reentrenaron las 12 corridas con otra configuración y se mantiene la base. Si se repitiera, la configuración sugerida es lr 3e-4.

## Limitaciones

- **La verdad es ERA5, no el clima medido.** ERA5 sobreestima o subestima la lluvia de las estaciones entre 15 % y 60 %. Un modelo que aprende de ERA5 aprende también sus errores. La evaluación contra las estaciones de ANACAFE y el afinado con ellas siguen pendientes.
- **Mejor época 2 en casi todas las corridas** (sobreajuste confirmado en las curvas). El barrido de arriba lo retrasa pero no mejora el resultado. No se probaron otras ventanas de entrada ni arquitecturas más pequeñas.
- **Un periodo de test (2021–2026) con ~2,100 ventanas por región**, muy autocorrelacionadas: el número efectivo de muestras independientes es mucho menor.
- **No se comparó contra el pronóstico numérico de Open-Meteo** de forma justa: contra ERA5 no es un techo limpio (ver `data/README.md`).
- Los hiperparámetros de las tres arquitecturas se fijaron a priori y no se ajustaron; el tamaño (106 mil a 209 mil parámetros) difiere entre ellas.

## Evaluación contra las estaciones de ANACAFE (clima medido)

Hasta aquí todo se midió contra ERA5. Con las 6 estaciones como verdad (`python -m src.forecast.station_eval`; `experiments/forecast/station_eval.csv`), con periodo de ajuste 2022–2024 y **test desde el 1-jul-2025** (380–480 días por región, el mismo conjunto de filas para todos los predictores), MAE medio de las 6 regiones. Los modelos reciben ERA5 como entrada (sin afinar). `+sesgo` suma el sesgo medio contra la estación calculado solo en 2022–2024.

| Variable | h | Persist. estación | Clima ERA5 | Numérico (Open-Meteo) | Modelo GRU | GRU + sesgo |
|---|---|---|---|---|---|---|
| Temperatura (°C) | 1 | **0.89** | 1.40 | 1.32 | 1.13 | 1.09 |
| | 7 | 1.33 | 1.40 | 1.39 | **1.31** | 1.34 |
| | 14 | 1.44 | 1.40 | — | **1.35** | 1.39 |
| Humedad (pts) | 1 | **3.61** | 6.74 | 6.21 | 6.61 | 4.53 |
| | 7 | 5.61 | 6.71 | 5.63 | 6.66 | **5.27** |
| | 14 | 5.88 | 6.71 | — | 6.63 | **5.53** |
| Lluvia (mm/día) | 1 | 6.77 | 6.38 | 6.31 | **5.98** | 6.79 |
| | 7 | 8.17 | 6.44 | 7.10 | 6.57 | 7.69 |
| | 14 | 8.01 | **6.41** | — | 6.73 | 7.89 |

Lo que dicen los datos:

1. **El error real es mayor que el medido contra ERA5.** Contra ERA5 el GRU tenía 0.52 °C a 1 día; contra la estación tiene 1.13 °C. ERA5 mismo se aparta ~1 °C de la estación, y eso es un piso que ningún modelo entrenado con ERA5 puede bajar.
2. **Humedad: la ganancia principal es corregir el sesgo.** ERA5 subestima la humedad medida entre 2 y 10 puntos; sumar el sesgo baja el error de 6.6 a 4.5 puntos a 1 día. Con esa corrección los modelos superan a todos los baselines a 3–14 días (a 7 días: 5.27 frente a 5.61 de la persistencia de estación y 5.69 de la climatología corregida), por 3–8 %.
3. **Temperatura: poca habilidad más allá del primer día.** A 7–14 días los modelos (1.31–1.35 °C) casi empatan con la climatología (1.40) y la persistencia de la estación (1.33–1.44). La corrección de sesgo no ayuda (la altitud ya corrige lo grueso).
4. **Lluvia: el modelo crudo gana a 1 día (5.98 frente a 6.3–6.8) y empata con la climatología desde ~7 días.** La corrección de sesgo **aditiva empeora la lluvia** (5.98 → 6.79): es el tipo de corrección equivocado para una variable asimétrica con muchos días secos.
5. **El pronóstico numérico de Open-Meteo no es un techo tampoco contra las estaciones**: a 1 día pierde frente a los modelos en las tres variables (temperatura 1.32 frente a 1.13; lluvia 6.31 frente a 5.98) y frente a la persistencia de la estación en temperatura y humedad. Su columna `+sesgo` no es fiable (el sesgo se estimó con solo 2024, el único año en que existe).
6. **La persistencia de la propia estación es el rival más duro a 1–3 días**, pero requiere que el productor tenga estación; el sistema debe poder funcionar sin ella.

**Advertencia de despliegue:** ERA5 llega con ~5 días de retraso. En operación los últimos días de la ventana de entrada no existirán y habría que completarlos con otra fuente (la API de pronóstico con `past_days`), cuya distribución no es la de entrenamiento. Estas cifras suponen la ventana completa de ERA5 y por eso son optimistas.

**Qué sigue (afinado):** estos resultados fijan el control justo. Un modelo afinado con las estaciones debe superar a "preentrenado + sesgo" (4.5 en humedad a 1 día, 1.09 en temperatura) y a la persistencia de estación, y no a la versión cruda; para la lluvia, el control es el modelo crudo.

## Afinado con las estaciones (transfer learning, Lab 6) — Fase 3

`python -m src.forecast.finetune`; `experiments/forecast/station_eval_finetune.csv` y `finetune_log.json`. Se parte de los modelos preentrenados con ERA5 (GRU y Transformer, 4 semillas) y se reentrenan con la **misma entrada ERA5 pero la estación como objetivo**, con dos estrategias: `head` (solo la cabeza lineal; feature extraction) y `full` (todo, learning rate 1e-4). Ajuste: 6,138 ventanas con objetivo en 2022–2024 (97.6 % de los días de estación con dato); parada con 995 ventanas de validación de 2025-01 a 2025-06; **test: día objetivo desde 2025-07-01**, nunca usado para ajustar. La pérdida de validación contra la estación bajó de ~0.55 a ~0.39 en las 16 corridas.

MAE contra la estación, media de las 6 regiones y de 4 semillas (± desviación entre semillas). El **control justo** es "preentrenado + corrección de sesgo" (y, para la lluvia, el modelo crudo, porque el sesgo aditivo la empeora):

| Variable | h | Control | Afinado `head` | Afinado `head` + sesgo | Mejor baseline sin afinar |
|---|---|---|---|---|---|
| Lluvia (mm/día) | 1 | 5.98 ± 0.09 (GRU crudo) | **5.28 ± 0.02** | — | 6.31 (numérico) |
| | 7 | 6.57 ± 0.04 | **5.63 ± 0.03** | — | 6.44 (clima) |
| | 14 | 6.73 ± 0.14 | **5.66 ± 0.03** | — | 6.41 (clima) |
| Humedad (pts) | 1 | 4.53 ± 0.06 | 4.53 ± 0.07 | **4.05 ± 0.03** | 3.61 (persist. estación) |
| | 7 | 5.27 ± 0.10 | 5.27 ± 0.06 | **4.79 ± 0.03** | 5.61 (persist. estación) |
| | 14 | 5.53 ± 0.11 | 5.46 ± 0.08 | **4.97 ± 0.05** | 5.67 (clima + sesgo) |
| Temperatura (°C) | 1 | 1.09 ± 0.01 | 1.16 ± 0.03 | 1.02 ± 0.01 | **0.89** (persist. estación) |
| | 7 | 1.34 ± 0.01 | 1.34 ± 0.03 | **1.24 ± 0.02** | 1.33 (persist. estación) |
| | 14 | 1.39 ± 0.03 | 1.37 ± 0.04 | **1.26 ± 0.02** | 1.40 (clima) |

(Cifras del GRU; el Transformer da valores equivalentes: p. ej. lluvia 5.26 / 5.59 / 5.66 con `head`.)

Lo que se puede afirmar:
1. **Lluvia: el afinado es la mejora más clara** (−12 %, −14 % y −16 % a 1, 7 y 14 días frente al modelo crudo; supera además a la climatología, a la persistencia y al pronóstico numérico). Mejora en las 6 regiones a 7 días (de +5 % en Huehuetenango a +26 % en Antigua).
2. **Humedad y temperatura: afinar por sí solo no supera al control** (la humedad empata exactamente: 4.53 frente a 4.53 a 1 día; la temperatura a 1 día es peor, 1.16 frente a 1.09). La mejora aparece al **sumar después una corrección de sesgo residual** calculada en el periodo de ajuste: ~10 % en humedad y 7–9 % en temperatura a 7–14 días, en 5 de 6 regiones (la excepción es Huehuetenango). Por qué ese segundo paso ayuda si el modelo ya se entrenó con esos datos no se investigó.
3. **`head` y `full` rinden igual** (lluvia a 7 días: 5.63 frente a 5.80 en el GRU; 5.59 frente a 5.57 en el Transformer): con ~6,000 ventanas basta reentrenar la cabeza, como predice la guía de selección de la semana 9.
4. **A 1 día, la persistencia de la propia estación sigue siendo la mejor** en temperatura (0.89) y humedad (3.61), pero exige tener estación.

Limitaciones del afinado: (a) **el ajuste y el test usan las mismas 6 estaciones** (en periodos distintos); la prueba en una finca nueva está en la sección siguiente (dejando una estación fuera); (b) un solo periodo de test de ~16 meses (muy autocorrelacionado); (c) los modelos reciben ERA5 de entrada, que en operación tiene ~5 días de retraso (ver advertencia arriba); (d) el afinado aprende también los errores de las estaciones que el control de calidad no detectó.

## Finca nueva: afinado dejando una estación fuera (`src/forecast/loso.py`)

Para cada región se afina el modelo (`head`, 4 semillas por familia) solo con las otras 5 estaciones y se evalúa en la
estación que no vio, con el mismo test (desde 2025-07-01). MAE medio de las 6 regiones, GRU (el Transformer da
valores equivalentes; resultados completos en `experiments/forecast/loso_eval.csv`):

| Variable | h | Preentrenado (ERA5) | Afinado con otras 5 | Cambio | Clima | Numérico |
|---|---|---|---|---|---|---|
| Lluvia (mm/día) | 1 | 5.96 | **5.32** | −11 % | 6.36 | 6.29 |
| | 7 | 6.49 | **5.58** | −14 % | 6.37 | 7.03 |
| | 14 | 6.72 | **5.70** | −15 % | 6.40 | — |
| Humedad (pts) | 1 | 6.64 | **5.17** | −22 % | 6.77 | 6.24 |
| | 7 | 6.71 | 5.92 | −12 % | 6.75 | **5.65** |
| | 14 | 6.68 | **6.11** | −9 % | 6.76 | — |
| Temperatura (°C) | 1 | **1.14** | 1.34 | +18 % | 1.40 | 1.32 |
| | 7 | **1.32** | 1.49 | +13 % | 1.40 | 1.39 |
| | 14 | **1.35** | 1.51 | +12 % | 1.40 | — |

1. **La mejora de lluvia generaliza a una finca sin estación** (−11 % a −15 %, igual que con las estaciones conocidas) y es mejor en 6 de 6 regiones a 3, 7 y 14 días.
2. **Humedad: también mejora** (mejor en 5 de 6 regiones), pero a 7 días el pronóstico numérico de Open-Meteo todavía gana (5.65 frente a 5.92).
3. **Temperatura: el afinado empeora** (solo mejora en 2 de 6 regiones). Es razonable: el sesgo de temperatura depende de la altitud y cada estación tiene la suya, así que el sesgo promedio de otras fincas no sirve. Para una finca nueva se usa el modelo preentrenado, que a 7–14 días queda ligeramente por debajo de la climatología y del pronóstico numérico.
4. **La corrección de sesgo con las otras estaciones no ayuda**: empeora la lluvia (p. ej. 7.66 frente a 6.49 a 7 días) y no mejora la temperatura. Con el sesgo de la propia estación (referencia, no disponible en una finca nueva) la humedad llega a 5.30 a 7 días: **tener estación propia vale más que cualquier ajuste con las ajenas**.

Con esto el artefacto (`src/forecast/export.py`) usa el modelo afinado para lluvia y humedad y el preentrenado para temperatura; la elección sale de este cuadro, no de una decisión a mano. Limitaciones: seguimos con un solo periodo de test de ~16 meses y solo 6 fincas, todas dentro de las regiones cafetaleras de ANACAFE.

## Retraso de ERA5 en operación (`src/forecast/latency.py`)

ERA5 llega con 5–8 días de retraso, así que al pronosticar "hoy" los últimos días de la entrada no existen. Se midió repitiendo el último valor conocido en esos días (MAE contra ERA5, test, media de regiones y semillas del GRU):

| Variable | h | k = 0 | k = 5 días | k = 8 días |
|---|---|---|---|---|
| Temperatura (°C) | 1 | 0.52 | 0.90 (+73 %) | 0.93 (+79 %) |
| | 7 | 0.85 | 0.86 (+2 %) | 0.88 (+4 %) |
| | 14 | 0.89 | 0.90 (+1 %) | 0.91 (+2 %) |
| Humedad (pts) | 1 | 3.61 | 5.56 (+54 %) | 5.91 (+64 %) |
| | 7 | 5.14 | 5.33 (+4 %) | 5.37 (+4.5 %) |
| | 14 | 5.34 | 5.41 (+1 %) | 5.43 (+2 %) |
| Lluvia (mm/día) | 1 | 3.76 | 4.55 (+21 %) | 4.68 (+25 %) |
| | 7 | 4.31 | 4.36 (+1 %) | 4.36 (+1 %) |

La API de pronóstico de Open-Meteo (`past_days`) sí trae los días recientes, pero **no es un buen relleno**: contra ERA5 en los mismos 38 días, la temperatura media difiere 0.95 °C de error medio (la variación natural es 0.72 °C), la mínima tiene correlación 0.42 y la lluvia 0.52. Decisión: relleno por persistencia y, con retraso, los horizontes menores a 7 días (`min_horizon_with_gap`, calculado con tolerancia de 5 % a k = 5) se marcan como poco confiables. Limitación: es el peor caso sin datos nuevos; no se probó un relleno mejor (p. ej. entrenar con entradas incompletas), que podría recuperar parte del error a 1–3 días.

### Repetición con otro corte de fechas

Para ver si lo anterior depende del periodo de test, se repitió la validación en finca nueva con un corte distinto
(`stations_alt.corte2` en `configs/forecast.yaml`): ajuste 2022–2023 (un año menos), validación en 2024-01 a 2024-06 y
test desde 2024-07-01 (762–845 días por región y variable, frente a ~16 meses antes). Solo GRU, 4 semillas
(`experiments/forecast/loso_eval_corte2.csv`):

| Variable | h | Preentrenado | Afinado con otras 5 | Cambio | Regiones donde mejora | Numérico |
|---|---|---|---|---|---|---|
| Lluvia (mm/día) | 1 | 6.05 | **5.32** | −12 % | 6 de 6 | 5.98 |
| | 7 | 6.32 | **5.51** | −13 % | 6 de 6 | 6.49 |
| | 14 | 6.44 | **5.56** | −14 % | 6 de 6 | — |
| Humedad (pts) | 1 | 6.76 | **5.33** | −21 % | 5 de 6 | 7.51 |
| | 7 | 7.02 | **5.91** | −16 % | 5 de 6 | 6.17 |
| | 14 | 7.00 | **6.05** | −14 % | 5 de 6 | — |
| Temperatura (°C) | 1 | **1.15** | 1.42 | +23 % | 2 de 6 | 1.33 |
| | 7 | **1.31** | 1.56 | +19 % | 2 de 6 | 1.32 |
| | 14 | **1.32** | 1.56 | +18 % | 2 de 6 | — |

Las tres conclusiones se mantienen con el otro corte: lluvia y humedad mejoran con el afinado, la temperatura empeora y se
usa el modelo preentrenado. Dos diferencias: la mejora de humedad es mayor (a 7 días el afinado ya supera al pronóstico
numérico, que antes ganaba), y la climatología queda cerca del modelo preentrenado en temperatura (1.32 frente a 1.31 a 14
días), así que ahí el modelo casi no aporta. Esto reduce la preocupación por el periodo de test, no la elimina: los dos
cortes comparten estaciones, años y el mismo preentrenamiento.

### Entrenar con entradas incompletas (probado y descartado)

Se probó entrenar el GRU repitiendo el último día conocido en los últimos 1–8 días de la entrada en la mitad de las
muestras, para reducir el costo del retraso de ERA5 a 1–3 días (4 semillas, MAE contra ERA5 en el test). Con 5 días de
retraso y 1 día de horizonte el error bajó 5–8 % (temperatura 0.90 → 0.84 °C, humedad 5.56 → 5.13 pts, lluvia 4.55 →
4.31 mm/día), pero con la entrada completa el modelo empeoró (+3 % en lluvia, +10 % en humedad, +16 % en temperatura) y de 3
días en adelante no cambió. El error relativo con retraso siguió siendo grande a 1 día (+29 % a +39 %), así que el
horizonte mínimo confiable seguía siendo 7. **Se descartó y se eliminó el código**: la limitación a 1–3 días es de
información (los últimos días no existen), no algo que se corrija entrenando mejor. Estas cifras quedan aquí como
evidencia del intento; ya no se pueden reproducir desde el repositorio.

## Artefacto e inferencia

`python -m src.forecast.export` escribe `models/forecast/artifact.json` (pesos por variable, normalizadores por región, horizonte mínimo con retraso) y `src/forecast/infer.py` (`WeatherForecaster`) lo usa. `python -m src.forecast.check_infer` lo verifica con los pesos reales: coincide con el camino de evaluación (diferencia 0) y marca el retraso correctamente. Una finca nueva usa el normalizador de la región más cercana (aproximación).

## Implicación para el sistema

Para la red bayesiana importa sobre todo el **clima de los próximos días** (humedad y lluvia, que favorecen a la roya). Ahí el modelo aporta algo útil hasta ~3 días y poco más allá. Alimentar a la red con un pronóstico a 14 días como si fuera confiable no estaría justificado: la evidencia climática futura debe llevar su incertidumbre (por ejemplo, mezclando con la climatología a medida que sube el horizonte).

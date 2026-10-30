# Resultados de visión (Fase 2)

Fuente de verdad: `experiments/vision/<fecha>_<nombre>/results.json` y `metrics.csv`. Configuración en `configs/vision_*.yaml`.
Todas las corridas: 256×256 px, AdamW, 15 épocas máximo con early stopping (paciencia 5), pesos de clase inversos a la frecuencia, semilla 42, **una sola corrida por configuración**.
Los tests son pequeños (BRACOL 239 imágenes, 33 de roya severa; RoCoLe 218 imágenes, 36 de roya severa): las diferencias de pocos puntos de F1 caen dentro del ruido.

## F1 macro (4 clases) en cada test

Baselines: clase mayoritaria 0.164 (BRACOL) y 0.168 (RoCoLe); azar uniforme 0.231 y 0.279.

| Corrida | Entrenada con | Test BRACOL | Test RoCoLe |
|---|---|---|---|
| CNN pequeña desde cero | BRACOL | 0.862 | 0.203 |
| ResNet-18, solo cabeza | BRACOL | 0.713 | 0.114 |
| ResNet-18, última capa + cabeza | BRACOL | 0.865 | 0.159 |
| ResNet-18, LR diferencial | BRACOL | **0.928** | 0.126 |
| EfficientNet-B0, LR diferencial | BRACOL | 0.918 | 0.241 |
| ResNet-18, LR diferencial | RoCoLe | 0.371 | 0.711 |
| **ResNet-18, LR diferencial** | **BRACOL + RoCoLe** | **0.923** | **0.727** |

## Hallazgos

1. **Cambio de dominio severo.** Un modelo entrenado solo con BRACOL (hojas sobre fondo blanco) cae a F1 0.11–0.24 en fotos de campo de RoCoLe, **por debajo del azar**; el entrenado solo con RoCoLe cae a 0.37 en BRACOL.
2. **Entrenar con ambos datasets lo resuelve** sin costo en BRACOL (0.923 frente a 0.928) y con F1 0.727 en RoCoLe.
3. **El orden de las estrategias coincide con el Lab 6:** solo cabeza < parcial < completo con LR diferencial.
4. **ResNet-18 y EfficientNet-B0 empatan en BRACOL** (0.928 y 0.918, dentro del ruido). EfficientNet generaliza algo mejor a RoCoLe sin entrenar con él (0.241 frente a 0.126), pero sigue cerca del azar.
5. **Una CNN pequeña desde cero llega a 0.862 en BRACOL**: señal de que ese dataset es fácil (hojas centradas sobre fondo blanco) y de que el modelo podría usar pistas de adquisición. Hay que validarlo con Grad-CAM.

## Detección de roya (leve o severa) frente a no roya (sana u otra afección), con IC 95 % de Wilson

Es la medida que usa la alarma preventiva (`docs/decision_umbral_roya.md`).

| Modelo | Test | Sensibilidad | Especificidad |
|---|---|---|---|
| ResNet-18, solo BRACOL | BRACOL | 0.96 [0.90–0.99] (n=81) | 0.99 [0.96–1.00] (n=158) |
| ResNet-18, solo BRACOL | RoCoLe | 0.80 [0.70–0.87] (n=84) | **0.54** [0.46–0.63] (n=134) |
| **ResNet-18, combinado** | BRACOL | 0.96 [0.90–0.99] | 0.96 [0.92–0.98] |
| **ResNet-18, combinado** | RoCoLe | **0.71** [0.61–0.80] | **0.94** [0.89–0.97] |

En fotos de campo el modelo combinado detecta ~7 de cada 10 hojas con roya y casi no da falsas alarmas. Esos valores son los que entran como `P(O | L)` en la red bayesiana.

## Segunda tanda: 4 semillas por configuración (modelos entrenados con BRACOL + RoCoLe)

Media ± desviación entre semillas (42, 1, 2, 3); se regenera con `python -m src.vision.summarize` (`experiments/vision/summary.csv`).

| Modelo | F1 macro BRACOL | F1 macro RoCoLe | Roya sev. (recall) RoCoLe | Detección de roya en RoCoLe: sensibilidad | especificidad |
|---|---|---|---|---|---|
| ResNet-18, LR diferencial | 0.936 ± 0.011 | 0.719 ± 0.015 | 0.625 ± 0.036 | 0.783 ± 0.088 | 0.929 ± 0.037 |
| EfficientNet-B0, LR diferencial | 0.917 ± 0.028 | 0.716 ± 0.030 | 0.736 ± 0.066 | 0.887 ± 0.023 | 0.896 ± 0.021 |

- **Empate en F1:** las diferencias (0.019 en BRACOL, 0.003 en RoCoLe) son menores o comparables a la desviación entre semillas. No hay evidencia de que un backbone supere al otro.
- **Detección de roya en fotos de campo:** EfficientNet-B0 detecta más (0.887 frente a 0.783) con algo menos de especificidad (0.896 frente a 0.929) y mucha menos variabilidad entre semillas.
- **La cifra de 0.71 de sensibilidad citada arriba venía de una sola semilla** (la 42); con 4 semillas ResNet-18 promedia 0.783 con una desviación de 0.088, es decir, esa corrida estaba en el extremo bajo. Con una sola semilla no se debe reportar esa cifra.
- Una semilla de EfficientNet-B0 (la 3) fue un valor atípico: mejor época 3 y F1 0.877 en BRACOL.
- La selección del modelo no se hace con estos tests (son la evaluación final): se decide con la validación (ver «Selección final» más abajo).

## Selección final, calibración y Grad-CAM (cierre de la Fase 2)

Código: `python -m src.vision.finalize` (calibra, compara candidatos, exporta) y `python -m src.vision.validate_explanations`. Resultados en `experiments/vision/final_comparison.csv`, `confusion_hoja.json`, `gradcam_summary.json` y `gradcam_validation.csv`.

### Candidatos y regla de selección

Regla fijada antes de ver el test: **menor NLL calibrado en la validación conjunta** (BRACOL + RoCoLe). A cada miembro se le ajusta una temperatura sobre esa validación; la probabilidad del candidato es el promedio de sus miembros calibrados.

| Candidato | Miembros | NLL val (calibrado) | F1 val | F1 test BRACOL | F1 test RoCoLe |
|---|---|---|---|---|---|
| A. ResNet-18 combinado, semilla 42 | 1 | 0.350 | 0.841 | 0.923 | 0.727 |
| B. Ensamble solo-BRACOL + solo-RoCoLe (decisión D3) | 2 | 0.954 | 0.711 | 0.702 | 0.642 |
| C. Ensamble de 4 semillas de ResNet-18 combinado | 4 | 0.295 | 0.879 | **0.958** | 0.744 |
| **D. Ensamble de 4 ResNet-18 + 4 EfficientNet-B0 combinados** | 8 | **0.286** | **0.886** | 0.949 | **0.787** |
| E. EfficientNet-B0 combinado, semilla 42 | 1 | 0.341 | 0.858 | 0.928 | 0.720 |

- **Elegido por la regla: D.** C queda muy cerca (NLL 0.295 frente a 0.286; en test C gana en BRACOL y D en RoCoLe): la diferencia entre ambos no es concluyente. D cuesta 8 pasadas por foto (166–200 ms por imagen en CPU, medido) y sus pesos no se versionan.
- **Decisión D3 resuelta: el ensamble de modelos específicos de cada dataset (B) es claramente peor** (F1 0.702 en BRACOL). El modelo solo-BRACOL necesita una temperatura de 11.7 (está muy sobreconfiado fuera de su dominio) y el solo-RoCoLe 2.6; los combinados, entre 1.07 y 1.80.
- Una ventaja menor de los ensambles en NLL puede deberse a que tienen más parámetros ajustados sobre la validación (una temperatura por miembro).

### Calibración: no se demostró mejora en el test

| Candidato | ECE sin calibrar → calibrado, BRACOL | ECE sin calibrar → calibrado, RoCoLe |
|---|---|---|
| A | 0.048 → 0.055 | 0.053 → 0.039 |
| D (elegido) | 0.042 → 0.071 | 0.048 → 0.074 |
| E | 0.046 → 0.047 | 0.093 → 0.053 |

La temperatura mejora la NLL de validación por construcción, pero el ECE del test no mejora de forma consistente: baja para los modelos individuales de RoCoLe y **empeora ligeramente para el ensamble** (probablemente porque el promedio de miembros ya calibrados queda subconfiado). Con ~240 imágenes el ECE tiene un ruido de unos 0.03, así que estas diferencias no son concluyentes. Pendiente: probar una sola temperatura sobre el ensamble completo, evaluándola con validación cruzada dentro de la validación y no con el test.

### Detección de roya del modelo elegido (matriz de confusión por hoja)

`experiments/vision/confusion_hoja.json` guarda las matrices de 4 clases (alimentan `P(O|L)` de la red bayesiana).

| Test | Sensibilidad de roya (leve o severa) | Especificidad | Recall de roya severa |
|---|---|---|---|
| BRACOL | 1.000 [0.95–1.00] (n=81) | 0.981 [0.95–0.99] (n=158) | 0.909 (n=33) |
| RoCoLe (campo) | 0.881 [0.79–0.93] (n=84) | 0.925 [0.87–0.96] (n=134) | 0.722 (n=36) |

En fotos de campo la roya severa es la clase más débil: 9 de 36 hojas se confunden con roya leve.

### Grad-CAM sobre los modelos reales (32 imágenes por dataset, 8 por clase, clase predicha)

| Modelo, dataset | Deletion (↓) CAM / centro / azar | Insertion (↑) CAM / centro / azar | CAM mejor que el centro (deletion / insertion) |
|---|---|---|---|
| ResNet-18, BRACOL | 0.319 / 0.358 / 0.534 | 0.849 / 0.827 / 0.675 | 23/32 (p=0.02) · 23/32 (p=0.02) |
| ResNet-18, RoCoLe | 0.345 / 0.402 / 0.432 | 0.700 / 0.660 / 0.538 | 22/32 (p=0.05) · 23/32 (p=0.02) |
| EfficientNet-B0, BRACOL | 0.310 / 0.423 / 0.630 | 0.892 / 0.820 / 0.720 | 26/32 (p=0.001) · 27/32 (p<0.001) |
| EfficientNet-B0, RoCoLe | 0.299 / 0.384 / 0.486 | 0.749 / 0.713 / 0.468 | 27/32 (p<0.001) · 24/32 (p=0.007) |

("centro" = un mapa fijo en forma de mancha centrada, el control para la composición de las fotos; p = prueba binomial de dos colas contra el 50 %.)

1. **Pasa la prueba de aleatorización.** Al reiniciar los pesos desde la capa de salida, la correlación de Spearman con el mapa original cae a ≈ 0 en cuanto se reinicia el último bloque (ResNet-18: −0.10 y 0.07 en `layer4`; EfficientNet-B0: entre −0.04 y 0.16 desde el clasificador). El mapa depende de lo aprendido. Matiz: en ResNet-18, reiniciar solo la capa `fc` deja una correlación de 0.66–0.70: buena parte del mapa viene de dónde se activa `layer4`, no de los pesos de clase.
2. **No mira el fondo.** En BRACOL el cociente entre la atención sobre el fondo claro y el área del fondo (50 % de la imagen) es 0.38 en ResNet-18 y 0.31 en EfficientNet-B0; ninguna imagen prefiere el fondo (heurístico de brillo y saturación, solo válido para BRACOL).
3. **Es informativo, pero gran parte de la ventaja sobre el azar la explica estar centrado.** El Grad-CAM supera claramente al orden aleatorio, pero un mapa fijo centrado ya explica entre el 34 % y el 87 % de esa mejora. Lo que el Grad-CAM agrega al centro es real (gana en 69–84 % de las imágenes) y modesto.
4. **Revisión visual** (`docs/figures/gradcam_ejemplos_resnet18_dlr_both.png`): en BRACOL los mapas son manchas anchas sobre la hoja, con el máximo cerca de la lesión en algunos casos (hoja con roya severa) pero **sin localizar pústulas individuales**; la resolución de Grad-CAM es de 8×8. En RoCoLe algunos errores (p. ej. roya severa → sana) atienden a otras hojas de la escena. **No se debe afirmar que el modelo "ve las pústulas"**; sí que usa la hoja y no el fondo.

### Verificación de punta a punta

`python -m src.vision.check_infer` pasa las imágenes de test por `infer.py` con el artefacto real: F1 macro de 0.956 en BRACOL (referencia 0.949) y 0.787 en RoCoLe (referencia 0.787); la diferencia de 0.006 se debe a usar float32 en vez de precisión mixta.

## Fotos no concluyentes (`python -m src.vision.abstention`)

El clasificador siempre responde una de las 4 clases, también ante una foto que no es una hoja. Se agregó una regla de **abstención** que marca una
foto como «no concluyente» (`usable = False` con su `reason` en `infer.py`) y la API no la cuenta como observación en la decisión. La regla se fijó con la
validación, antes de mirar el test: `min_confidence` es el percentil 10 de la confianza del ensamble en la validación conjunta (0.577) y `min_pixel_std`
es la mitad del menor contraste (desviación de gris) visto en validación y test (10.8 de 255). El 10 % es un parámetro de diseño. Resultados en
`experiments/vision/abstention.json`:

| Corte (percentil de validación) | Test BRACOL: cobertura / exactitud de lo conservado | Test RoCoLe: cobertura / exactitud de lo conservado |
|---|---|---|
| ninguno | 1.00 / 0.962 | 1.00 / 0.849 |
| 5 % (0.498) | 0.983 / 0.975 | 0.862 / 0.888 |
| **10 % (0.577, el usado)** | **0.958 / 0.983** | **0.803 / 0.914** |
| 20 % (0.694) | 0.891 / 0.986 | 0.716 / 0.936 |

- **Costo:** en fotos de campo (RoCoLe) la regla descarta ≈ 20 % de las hojas válidas; en BRACOL, ≈ 4 %. Lo que se conserva es más exacto (0.849 → 0.914 en RoCoLe).
- **Imágenes sintéticas que no son hojas** (ruido, verde y gris uniformes, tablero, degradado, manchas de color): las 6 quedan marcadas (confianza 0.28–0.51;
  las tres uniformes o de ruido también por poco contraste). Con 24 hojas reales del test, 23 se aceptan.
- **Lo que no demuestra:** no hay fotos reales de otras cosas (dedos, suelo, otras plantas) con las que medir un detector de «no es una hoja»; la regla solo descarta lo dudoso.
  Una foto de una hoja de otra planta podría salir con confianza alta. Sigue pendiente probar con fotos reales de productores.
- Volver a correr `python -m src.vision.finalize` reescribe el artefacto y borra estos dos valores: hay que repetir `python -m src.vision.abstention --write`.

## Limitaciones conocidas

- Cuatro semillas por configuración (media ± desviación en la segunda tanda); el candidato final es un ensamble de 8 modelos, no una sola corrida.
- La clase `otra afección` significa cosas distintas en cada dataset (ácaro rojo en RoCoLe; minador, phoma y cercospora en BRACOL): el modelo solo-RoCoLe tiene recall 0.00 de `otra` en BRACOL.
- La calibración por temperatura no demostró mejorar el error de calibración del test (ver arriba); las probabilidades del ensamble son razonables pero no están garantizadas como calibradas.
- Grad-CAM pasa la aleatorización de pesos y no mira el fondo, pero no localiza pústulas individuales (resolución 8×8) y un mapa centrado explica gran parte de su ventaja: no se presenta como diagnóstico.
- El clasificador no detecta "esto no es una hoja" de forma general: la regla de abstención (sección anterior) descarta lo dudoso y las imágenes sintéticas probadas, pero no se midió con fotos reales de otras cosas.
- Los tests son pequeños (239 y 218 imágenes) y de una sola partición; las fotos de los productores serán otro dominio (cámara, luz, fondo), así que estos números son una cota optimista.
- Pendiente: validar el mapa de atención contra las máscaras de lesión de BRACOL y probar con fotos reales de productores.

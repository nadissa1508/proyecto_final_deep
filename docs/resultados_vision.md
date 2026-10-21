# Resultados de visión (Fase 2, primera ablación)

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

## Limitaciones conocidas

- Una semilla por configuración; falta repetir con 3 y reportar media ± desviación.
- La clase `otra afección` significa cosas distintas en cada dataset (ácaro rojo en RoCoLe; minador, phoma y cercospora en BRACOL): el modelo solo-RoCoLe tiene recall 0.00 de `otra` en BRACOL.
- Las probabilidades aún no están calibradas, y falta validar con Grad-CAM que el modelo mira las lesiones y no el fondo.
- Las fotos de los productores serán otro dominio (cámara, luz, fondo): estos números son una cota optimista.

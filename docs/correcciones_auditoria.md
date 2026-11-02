# Correcciones de la auditoría

Un integrante del equipo auditó el proyecto con apoyo de Codex en la rama `auditoria` (commit `b7e77cc`, 2026-11-01). Este documento resume sus
hallazgos y qué se hizo con cada uno en `main`. La auditoría completa, sus scripts y sus evidencias (`docs/auditoria_revision.md`,
`docs/analisis_alarma_roya.md`, `docs/auditoria_evidencia/`, `experiments/fusion/sensibilidad_auditoria_*`) quedan en esa rama; no se unieron a `main`.

## Hallazgos corregidos

| Hallazgo | Corrección | Efecto en los resultados |
|---|---|---|
| 13 de las 1,096 ventanas de validación por región tenían sus primeros días objetivo en el periodo de entrenamiento | `split_windows` exige que toda la salida de validación caiga en validación (`src/forecast/data.py`, prueba en `tests/test_forecast_data.py`) | Ninguno: reentrenar el GRU (4 semillas) eligió las mismas épocas, así que los modelos y el MAE de test son idénticos (0.0 %). Ver `docs/resultados_forecast.md` y `experiments/forecast/val_fix_comparacion.csv` |
| La regla de fotos no concluyentes fijó el umbral de contraste mirando también el test | El contraste se calcula solo con validación: 13.021 en vez de 10.761 (`src/vision/abstention.py`, `models/vision/artifact.json`) | Ninguno en la cobertura de test: ambos cortes quedan por debajo del menor contraste de las hojas de test |
| `check_infer` del pronóstico calculaba la diferencia con la evaluación pero no fallaba | Falla si la diferencia supera 1e-4 (`src/forecast/check_infer.py`) | Con los pesos reales la diferencia es 0 |
| Un pedido de pronóstico con historia inválida o `as_of` anterior a la historia respondía 500 | Responde 422 con el motivo (`src/api/routers/forecast.py`, prueba en `tests/test_api_deep_learning.py`) | Solo la API |
| La tarjeta de decisión podía mostrar una respuesta tardía de otra parcela | Cada consulta lleva una versión y las respuestas viejas se ignoran; al cambiar de parcela, región, altitud, sombra o fotos se limpian los resultados (`frontend/src/components/RustDecisionCard.jsx`, prueba en `RustDecisionCard.test.jsx`) | Solo la interfaz |
| El relleno por persistencia estaba mal descrito (se rellena la historia de entrada, no los días pronosticados) | Textos corregidos en `README.md`, `docs/resultados_fusion.md` y el mensaje de la alerta (`src/fusion/favorability.py`) | Solo redacción |

## Hallazgos declarados (no se pueden corregir sin datos nuevos)

| Hallazgo | Cómo se declara |
|---|---|
| El test de visión se usó en decisiones intermedias (estrategia, mezcla de dominios) | «Test de desarrollo» reutilizado, no evaluación externa (`docs/resultados_vision.md`) |
| La elección de modelo por variable del pronóstico usa el test de la validación en finca nueva | «Conjunto de selección»; falta una evaluación independiente del sistema híbrido (`docs/resultados_forecast.md`) |
| El control «pronóstico perfecto» no es un techo matemático | Se llama «clima observado» y se explica por qué no descarta al pronosticador (`docs/resultados_fusion.md`) |
| La simulación usa 2018–2021, años usados como validación y para calibrar la evidencia | Se aclara que no son años totalmente no vistos (`docs/resultados_fusion.md`) |
| La simulación no modela la abstención; sus intervalos no son de campo | Limitación declarada (`docs/resultados_fusion.md`) |
| El modelo ordinal clima–roya es un índice exploratorio y la media nacional no identifica la distribución de fincas | Redacción corregida (`docs/resultados_fusion.md`) |
| La diferencia de F1 entre evaluación e inferencia se atribuía a la precisión mixta sin probarlo | Se presenta como hipótesis (`docs/resultados_vision.md`) |
| Referencias: la fila de la OIC enlazaba otro trabajo, faltaba el enlace de Pérez (2022) y se decía «matriz de BRACOL» cuando producción usa la combinada | Corregido (`docs/decision_umbral_roya.md`) |

## Lo que no se integró, y por qué

- **Capa de vigilancia `src/fusion/alerts.py` y el formulario de floración, inicio de lluvias y última aplicación.** El equipo eligió la alerta de
  `src/fusion/favorability.py` (`docs/decision_umbral_roya.md` §11). Con la ventana de 3 días y los horizontes marcados como poco confiables, el clima de
  esa capa quedaba siempre «indeterminado» en operación, y las fechas del formulario se pedían pero no cambiaban el aviso.
- **Mejoras de la rama que no eran estrictamente necesarias para esta entrega:** lectura de rutas de Windows desde Linux en `src/vision/preprocess.py`,
  pruebas extra de API y frontend y pruebas HTTP dentro de Docker. Siguen disponibles en la rama `auditoria`. (La respuesta 422 del pronóstico y la
  protección de la tarjeta sí se integraron; ver la primera tabla.)

## Evidencia de la auditoría que se usa en el informe

La auditoría probó la regla estricta de exigir roya **y** clima favorable a la vez para recomendar una acción: en la simulación bajaron las
aplicaciones, pero aumentaron la roya y las pérdidas, porque cuando el pronóstico falla el clima bloquea decisiones aunque las fotos muestren roya.
Esa evidencia (cifras en `experiments/fusion/sensibilidad_auditoria_*` de la rama `auditoria`) respalda que la alerta de `main` no dependa del clima.
Si la rama se borra en GitHub, esa evidencia deja de estar disponible.

# Integración (Fase 4): pronóstico + fotos → red bayesiana → MDP

Cifras de `experiments/fusion/` (regenerables, ver al final) y de `notebooks/10_integracion.ipynb`. **Léase con una advertencia:** la
relación entre el clima y el nivel de roya de la parcela, la mezcla de hojas y la dinámica del mundo simulado son **supuestos de diseño**
(`configs/fusion.yaml`); no existen datos de incidencia de roya emparejados con clima para medirlos. Lo que se reporta en la sección 4 es
el valor de la información *dentro de ese mundo*, no una medición de campo.

## 1. Qué se construyó

```
foto(s) de hoja ─► CNN (ensamble calibrado) ─► clase por hoja ─┐
                                                                ├─► red bayesiana extendida ─► b(R) ─► QMDP ─► acción
historia de ERA5 ─► GRU ─► pronóstico 14 días ─► evidencia T,H,P ┘                            (Q* del MDP heredado)
```

- `src/fusion/evidence.py`: la ventana pronosticada (7 o 14 días) se resume en temperatura media, humedad media y lluvia acumulada, y se
  compara con **lo habitual en esa finca y mes** (terciles de los propios pronósticos del modelo entre 1980 y 2017, por región y mes).
  Se reparte la probabilidad con una normal centrada en el valor pronosticado y la desviación del error de la ventana medida en 2018–2020
  (temperatura 0.67 / 0.59 °C, humedad 5.1 / 4.6 puntos, lluvia 1.00 / 0.95 en `log1p(mm)`, para 7 / 14 días). Es evidencia blanda.
- `src/fusion/network_ext.py`: red `T, H, P → R → Y` y `R → O_1…O_n` (una observación por hoja). `P(O | R) = Σ_L P(O | L) · P(L | R)` con
  `P(O | L)` la matriz de confusión **medida** en el test de visión (BRACOL + RoCoLe). Reutiliza `BayesianNetwork` y Variable Elimination
  heredados sin cambios.
- `src/fusion/qmdp.py`: `a* = argmax_a Σ_r b(r) · Q*((mes, r), a)` sobre el MDP heredado (por parcela).
- Salidas adicionales: incidencia de las hojas fotografiadas con intervalo de Wilson y corrección de Rogan–Gladen; con menos de 10 hojas la
  interfaz solo muestra el intervalo.

**Hallazgo sobre lo heredado.** La `P(R | T, H, P)` de la red heredada se estimó con 100 filas de un conjunto de la India: solo 4 de las 27
combinaciones de padres tienen datos y las otras 23 quedan en 1/3, 1/3, 1/3. El endpoint heredado devuelve 1/3 para un clima normal.
Conectarle el pronóstico tal cual no habría cambiado ninguna decisión, así que se reemplazó por un modelo ordinal explícito
(`P(R ≥ medio)` y `P(R = alto)` logísticas en `z = índice T + índice H + índice P − 3`), anclado a la incidencia nacional media de
4.85 % (mayo de 2022) con regiones entre 1.45 % y 8.35 % que reporta CEDICAFE. Es un supuesto, no un resultado aprendido.

## 2. Verificación técnica

- La red extendida coincide con `pgmpy` (prueba `test_pgmpy_agrees_with_variable_elimination`), y el camino rápido de producción
  coincide con la red con nodos de observación (`test_fast_belief_matches_the_network_with_observation_nodes`).
- QMDP con creencia concentrada reproduce la política de Value Iteration; con creencias mezcladas es la mezcla de los `Q*`.
- API: `POST /api/vision/predict`, `GET /api/forecast/{region}`, `POST /api/decision` con validación de tipo y tamaño; responden 503 con el
  motivo si faltan pesos. Probados con fotos reales del test y los modelos reales, en local y dentro del contenedor Docker (los endpoints
  heredados siguen respondiendo).
- Frontend: subida desde archivo (arrastrar y soltar o botón) con reducción a 1024 px, corrección de orientación y aviso de HEIC; botón de
  cámara del celular (`capture="environment"`); tarjetas de pronóstico y de decisión; mapa de atención por foto. Probado a mano con el
  backend real. **No** se probó la cámara en un celular real ni la red local. Las pruebas del frontend cubren la preparación de fotos (HEIC, reducción), la subida y el selector, no el aspecto.
- Pruebas automáticas: 65 de Python y 20 del frontend (vitest), ninguna falla (las de integración usan módulos sintéticos para no depender de pesos).

## 3. Sensibilidad de la decisión a los supuestos

Se movió cada supuesto de `configs/fusion.yaml`, uno a la vez, sobre 81 casos (3 climas × 3 meses × 9 combinaciones de fotos)
(`experiments/fusion/sensibilidad_resumen.csv`). La acción recomendada cambia en 0 % a 13.6 % de los casos:

| Supuesto movido | Casos con otra acción |
|---|---|
| Incidencia representativa por banda (3 / 8 % y 7 / 15 %), proporción de roya severa, otra afección | 0 % – 2.5 % |
| Matriz de confusión solo BRACOL / solo RoCoLe | 1.2 % / 4.9 % |
| Riesgo con clima neutro, pendiente del clima | 2.5 % – 7.4 % |
| Peso por hoja 0.5 / 0.25 (hojas no independientes) | 6.2 % / **13.6 %** |

El supuesto que más pesa es la independencia entre hojas (una foto de la misma planta no es una observación nueva).

## 4. Valor de la información en el mundo simulado

Mundo (`src/fusion/simulate.py`): clima real de ERA5 en 2018–2021 (años que ni el preentrenamiento ni el afinado usaron), roya con las
transiciones del MDP heredado modulada por el clima observado de las dos semanas (`exp(γ·z)`), 12 meses, 300 réplicas por región-año y
números aleatorios comunes entre escenarios. Media de 24 región-años; ± es el error estándar entre ellos. Ingreso anual en USD/ha
frente a no tener información (`B0`, ingreso medio 4,121.78):

| Escenario | Frente a B0 (USD/ha) | Fracción del techo |
|---|---|---|
| **A** techo: nivel de roya conocido | **+149.70** ± 4.57 | 100 % |
| B clima observado de las 2 semanas anteriores | −9.76 ± 3.17 | −7 % |
| C pronóstico (retraso de ERA5 de 0 / 5 días) | −2.56 ± 1.24 / −1.22 ± 0.68 | −2 % / −1 % |
| C0 pronóstico **perfecto** (control) | −7.51 ± 2.80 | −5 % |
| D0 solo fotos (10 hojas) | **+9.32** ± 2.20 | 6 % |
| D pronóstico + 10 hojas (retraso 0 / 5) | +2.10 ± 1.97 / +3.55 ± 1.87 | 1 % / 2 % |

Lo que se puede afirmar, y lo que no:

1. **Conocer el nivel de roya vale poco en dinero en este MDP**: el techo es 3.6 % del ingreso, porque el MDP heredado gana casi todo con
   la cosecha y las acciones cuestan decenas de dólares. Aun con información perfecta, lo que ganamos es acotado.
2. **Las fotos son lo único que aporta** de forma medible, y su valor crece con el número de hojas: con la CNN medida (matriz agrupada),
   10 hojas dan +7.9 y **30 hojas dan +44.5 USD/ha (30 % del techo)** en el mundo con `γ = 0.15` (`voi_sensibilidad.csv`).
3. **Si la CNN real es peor que la que cree la red, el valor se desploma**: con la matriz de RoCoLe como "verdad" del mundo y la red
   suponiendo la agrupada, 30 hojas dan solo +8.1. Una red demasiado confiada en su clasificador pierde valor.
4. **El clima no ayudó en ninguna configuración probada, ni siquiera con un pronóstico perfecto.** No es un problema del pronosticador. El
   diagnóstico `diagnose_climate.py` muestra por qué: en el mundo simulado, la correlación entre el clima de las dos semanas (z) y el nivel
   de roya del mes es −0.04, 0.03, 0.10 y 0.25 para `γ = 0, 0.15, 0.30, 0.60`: el nivel depende sobre todo del mes anterior. La red supone
   un vínculo mucho más fuerte (su `P(R = alto)` pasa de ~4 % a ~44 % entre `z = −3` y `z = +3`), y esa evidencia sobreconfiada confunde
   a QMDP. Cruzando la pendiente que cree la red (0.1, 0.25, 0.5) con la fuerza real (`γ` de 0 a 0.6), el valor del pronóstico perfecto
   fue negativo en las 12 combinaciones (de −2.4 a −14.6 USD/ha) y menos negativo cuanto menor la pendiente.
5. **No se puede concluir que el clima no importe para la roya**: la literatura y CEDICAFE lo consideran central, pero actúa sobre
   semanas y meses, y nuestro mundo simulado solo lo acopla dentro del mes. Lo que sí muestra la simulación es que, **sin datos de
   incidencia reales para calibrar el vínculo, la evidencia climática no se justifica como motor de la decisión**. Por eso, sin fotos,
   la respuesta de la API avisa que la decisión se apoya solo en el clima y que este aportó poco.

El control de 4 (`C0`) se añadió a propósito tras ver que `C` salía negativo: sin él habríamos atribuido el resultado al pronóstico.

## 5. Limitaciones

- El mundo simulado comparte el índice `z` con la red (con otra intensidad); mide valor *dentro de ese mundo*.
- Las hojas se tratan como independientes dado R; el exponente por hoja (`leaf_weight = 1`) es el valor más optimista.
- QMDP valora la información solo después de actuar: no sirve para decidir si conviene fotografiar más hojas.
- Las fotos de productores serán otro dominio. El clasificador no detecta "esto no es una hoja" de forma general: las fotos con poca confianza
  o sin contraste se marcan «no concluyentes» y no entran a la decisión (`docs/resultados_vision.md`), pero eso no se midió con fotos reales de otras cosas.
  La simulación de esta fase **no** modela esa abstención: supone que todas las hojas fotografiadas se usan.
- En operación, los primeros 5–8 días de la ventana de 14 siempre están rellenados por persistencia; no se probó usar solo los días 8–14.
- Con una incidencia real por región y mes (ANACAFE / CEDICAFE) se podría medir el vínculo clima–roya y calibrar la CPT; es la mejora
  con más valor esperado.

## 6. Regenerar

```bash
python -m src.fusion.build_evidence                      # terciles y error de la ventana (usa el pronosticador real)
python -m src.fusion.simulate --reps 300 --stale 0 5     # voi_resumen.csv
python -m src.fusion.simulate --reps 120 --stale 0 --sensitivity
python -m src.fusion.simulate --reps 150 --stale 0 --slopes 0.1 0.25 0.5 --gammas 0.0 0.15 0.3 0.6
python -m src.fusion.sensitivity                         # sensibilidad de la decisión a los supuestos
python -m src.fusion.diagnose_climate                    # correlación clima–roya del mundo simulado
```

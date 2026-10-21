# Decisión de diseño: de fotos de hojas a una alarma de roya en la parcela

**Estado:** decisión del equipo, pendiente de confirmar con CEDICAFE. **Fecha:** 2026-10-30.
**Afecta a:** `src/fusion/network_ext.py`, `src/fusion/qmdp.py`, el informe (sección de solución) y `PLAN.md` §1.2.

## 1. Decisión

1. **La alarma es preventiva:** se dispara por *evidencia de roya (aunque sea poca) + condiciones climáticas favorables + época del año*, y no por un umbral fijo de incidencia.
2. **La incidencia de la parcela se reporta, pero no dispara la alarma.** Se muestra como "% de hojas con roya", con intervalo de confianza y corregida por los errores del clasificador, que es la unidad que usa ANACAFE.
3. **El 5 % es un valor de referencia, no un umbral oficial.** Solo ayuda a definir los niveles bajo, medio y alto del modelo, y se somete a análisis de sensibilidad.
4. **Quien decide la acción es el MDP** (con la creencia sobre el nivel de roya), no una regla suelta sobre el conteo de hojas.

## 2. El problema: tres cosas distintas se llaman "porcentaje de roya"

| Medida | Qué mide | Dónde aparece |
|---|---|---|
| **Severidad de la hoja** | % del *área* de una hoja cubierta por lesiones | Etiquetas de RoCoLe (escala OIRSA) y BRACOL; lo que predice nuestra CNN (`leve` ≤ 5 %, `severa` > 5 %) |
| **Incidencia de la parcela** | % de las *hojas* de una parcela que tienen roya | Muestreo de ANACAFE/CoffeeCloud (30 plantas, 2 ramas); el "5 %" de Colombia y Ecuador; el promedio nacional de Guatemala (4.85 % en mayo de 2022) |
| **Área afectada** | % de las *hectáreas* con roya | Informe de la OIC (2013): 70 % del área de Guatemala |

Nuestras clases de imagen usan un 5 % que **coincide en número, pero no en concepto**, con el 5 % de incidencia de la literatura. Una hoja con más de 5 % de su área afectada no equivale a una parcela con más de 5 % de hojas afectadas.

## 3. Evidencia consultada

| Fuente | Qué dice | Cómo la usamos | Confianza |
|---|---|---|---|
| Pérez, N. (2022). *Prevención y control de la roya del cafeto*, CEDICAFE/ANACAFE (presentación) | El control químico "debe ser preventivo", cuando las incidencias son bajas y **al inicio de la época de lluvias**, cuando la epidemia empieza a crecer. Condiciones favorables: 21–25 °C, mojado de hojas, humedad relativa > 85 % y oscuridad. Incidencia nacional promedio en mayo de 2022: 4.85 %; por región, entre 1.45 % y 8.35 %. Existe un sistema de monitoreo y vigilancia desde 2012. Recomienda CoffeeCloud. **No da un umbral numérico.** | Base de la decisión | Alta (institución guatemalteca). https://www.anacafe.org/uploads/file/4d9dbb854bf940159d4489c200dc44da/Prevencion-Roya-Cafeto-2022.pdf
 |
| Cenicafé/FNC (Colombia, 2026) | Control químico con incidencia > 5 % (lotes susceptibles) o > 10 % (resistentes) | Valores de referencia para los niveles del modelo | Media (otro país) |
| Estudio universitario de Ecuador | Protector con incidencia < 5 %, sistémico por encima | Refuerza el 5 % como referencia | Media-baja (confirmar la referencia exacta antes de citar) | 


| Djuikem y Arino (2025), arXiv:2504.01055 | Modelo matemático: los productores suelen actuar solo cuando ven síntomas; un umbral **bajo** es mejor y uno alto desperdicia la oportunidad de control; con pocas esporas y R₀ bajo la infección puede extinguirse sola | Apoya actuar temprano | Media (teórico, no calibrado a Guatemala) |
| OIC (2013), ED 2157/13 | Epidemia 2012/13: 2.7 millones de sacos perdidos, unos US$500 millones, 70 % del área afectada en Guatemala | Contexto del problema | Alta | https://arxiv.org/pdf/2504.01055

**Fuentes descartadas:**
- El texto que se pegó como "manual de CATIE 2015" es el *Manual de producción y manejo de aves de patio* (Manual técnico n.º 128): trata de gallinas, no de roya.
- Un artículo de blog de una empresa de bioinsumos: promociona sus productos y no es citable. Sus datos (20–28 °C, humedad > 90 %) contradicen a CEDICAFE, que sí tiene autoridad.

## 4. Alternativas consideradas

| | A. Umbral por severidad de hoja | B. Alarma al llegar a 5 % de incidencia | **C. Prevención: roya + clima + época** |
|---|---|---|---|
| Idea | Una hoja con > 5 % del área afectada activa la alarma | Si ≥ 5 % de las hojas fotografiadas tienen roya, alarma | Roya detectada (aunque sea poca) con clima favorable en época de lluvias activa la alarma preventiva |
| A favor | Simple; coincide con las etiquetas | Misma unidad y número que la literatura | Es lo que dice CEDICAFE; funciona con pocas fotos; es consistente con el MDP |
| En contra | Mezcla severidad con incidencia; ninguna fuente lo respalda | El 5 % viene de Colombia y Ecuador, no de Guatemala; **con pocas fotos no se puede medir** (sección 6); CEDICAFE pide actuar *antes* de cualquier umbral | Puede recomendar fungicida de más |
| Mitigación | — | — | El MDP incluye el costo de cada acción, la época y la creencia; se valida por simulación |

**Elegimos C.** B se conserva como *información* para el productor, y A queda solo como clasificación de cada hoja.

## 5. Cómo funciona

El modelo tiene dos niveles, porque una foto describe una hoja y la decisión es sobre una parcela:

- **R** (nivel de roya de la parcela, 3 valores) se define por la incidencia real *p*: **bajo** *p* < 5 %, **medio** 5 % ≤ *p* < 10 %, **alto** *p* ≥ 10 %. Los cortes de 5 % y 10 % vienen de Cenicafé y son **parámetros del diseño**, no umbrales oficiales de Guatemala.
- **Hoja fotografiada:** su estado real *L* es `sana`, `roya` u `otra afección`. La probabilidad de que una hoja tenga roya dado *R* es la incidencia representativa de la banda (por ejemplo 2 %, 7 % y 15 %). Es un supuesto que se documenta y se prueba en el análisis de sensibilidad.
- **Clasificador:** lo que devuelve la CNN, *O*, depende del estado real: *P(O | L)* es la **matriz de confusión medida en el test de BRACOL**. Así sus errores se tienen en cuenta.
- **Nodo de la red bayesiana:** *P(O | R) = Σ_L P(O | L) · P(L | R)*. Cada hoja es una observación, con el supuesto (aproximado) de independencia dado *R*. La red ya condiciona *R* con temperatura, humedad y lluvia, así que **el clima entra solo**.
- **Salidas:**
  1. la creencia *b(R)*;
  2. la incidencia estimada, con intervalo de Wilson y corrección de Rogan–Gladen: *p̂ = (aparente + especificidad − 1) / (sensibilidad + especificidad − 1)*;
  3. la acción de QMDP: *argmax_a Σ_s b(s) · Q\*(s, a)*.
- **Época:** el MDP ya hace más probable el empeoramiento de la roya en la temporada de lluvias (mayo a octubre), por eso la política recomienda cobre preventivo en esos meses.
- **Mínimo de hojas:** la app pedirá al menos 10 (ideal 20–30), de plantas distintas.

Mejora pendiente para la Fase 4: los umbrales climáticos de la red heredada (temperatura 18/24 °C, humedad 60/80 %) no coinciden con los de CEDICAFE (21–25 °C, humedad > 85 %). Hay que rediscretizarlos, y las estaciones de ANACAFE (que miden humedad y punto de rocío) permiten construir un índice de favorabilidad.

## 6. Por qué no un 5 % duro: la estadística

Probabilidad de ver **al menos una hoja con roya** si la incidencia real es *p* y se fotografían *n* hojas, con un clasificador perfecto, 1 − (1 − *p*)ⁿ:

| Hojas (*n*) | *p* = 2 % | *p* = 5 % | *p* = 10 % | *p* = 20 % |
|---|---|---|---|---|
| 5 | 10 % | 23 % | 41 % | 67 % |
| 10 | 18 % | 40 % | 65 % | 89 % |
| 20 | 33 % | 64 % | 88 % | 99 % |
| 30 | 45 % | 79 % | 96 % | 100 % |
| 60 | 70 % | 95 % | 100 % | 100 % |

Si se ven **0 hojas** con roya en *n* hojas, la incidencia real podría ser todavía tan alta como (límite superior de Wilson, 95 %): 43 % con 5 hojas, 28 % con 10, 16 % con 20, 11 % con 30 y 6 % con 60. **Hacen falta unas 73 hojas sin roya para asegurar que la incidencia sea menor que 5 %.** Y si se ve 1 hoja de 10, el intervalo del 95 % va de 1.8 % a 40 %.

Conclusión: con las 10 a 30 fotos que razonablemente toma un productor **no se puede resolver un umbral de 5 %**. ANACAFE puede hacerlo porque muestrea 30 plantas × 2 ramas (cientos de hojas). Lo que sí es medible con pocas fotos es **la presencia de roya**, que es justo el criterio preventivo. Y el clasificador no es perfecto: con una sensibilidad de 0.8 las probabilidades de la tabla bajan (con 30 hojas y *p* = 5 %, de 79 % a 71 %), por eso se usa la matriz de confusión real.

## 7. Límites y riesgos

- **No hay un umbral oficial confirmado.** La presentación de CEDICAFE no trae uno, pero no podemos afirmar que ANACAFE no tenga ninguno en otros documentos. Se le consultará.
- **Los cortes y las incidencias representativas son supuestos.** Se hará un análisis de sensibilidad con otros cortes (3 %/8 % y 7 %/15 %).
- **Las hojas de una misma planta no son independientes**, así que la creencia puede salir sobreconfiada. Mitigación: pedir plantas distintas y limitar cuánto pesa cada observación.
- **Las fotos de los productores no son las del dataset** (fondo, luz, cámara): el rendimiento real puede ser menor al medido.
- **Riesgo de aplicar fungicida de más** (costo y resistencia). El MDP incluye el costo de cada acción y la política depende de la época.
- **Los datos de incidencia reales** (sistema de monitoreo de ANACAFE) permitirían calibrar los cortes; se solicitan a CEDICAFE.

## 8. Cómo se validará

1. Simulación del valor de la información (PLAN Fase 4): comparar las reglas A, B y C en temporadas simuladas, midiendo el ingreso esperado y el número de aplicaciones.
2. Análisis de sensibilidad de los cortes de *R* y de la incidencia representativa.
3. Si CEDICAFE comparte la incidencia regional mensual, comparar con la creencia del sistema.

## 9. Texto sugerido para el informe

> Las fuentes de ANACAFE y la literatura miden la roya de dos formas: la *severidad* de una hoja (porcentaje de su área afectada) y la *incidencia* de una parcela (porcentaje de hojas con roya). Nuestro clasificador estima la primera a partir de una foto, pero la decisión de manejo depende de la segunda. CEDICAFE recomienda un control químico preventivo, aplicado cuando la incidencia aún es baja y al inicio de la época de lluvias, y no fija un umbral numérico. Además, con las pocas fotos que toma un productor no es posible distinguir de forma confiable una incidencia de 5 %: se necesitarían unas 73 hojas sanas para descartarla con 95 % de confianza. Por eso el sistema no aplica un umbral fijo: trata cada hoja como una observación imperfecta (con las tasas de error medidas del clasificador), combina esa evidencia con el clima pronosticado en una red bayesiana, y deja que el MDP decida la acción preventiva según el costo y la época. La incidencia estimada se reporta al productor, con su intervalo, como información.

## 10. Referencias

- Pérez, N. (2022, junio). *Prevención y control de la roya del cafeto* [presentación]. CEDICAFE, ANACAFE. **[completar enlace o nombre del archivo]**
- Cenicafé/FNC. (2026). Estado fitosanitario y nutricional, primer trimestre. https://www.agronegocios.co/comentarios/profesor-yarumo-401609/estado-fitosanitario-y-nutricional-primer-trimestre-2026-4346374
- Cenicafé. Infografía de control de la roya. https://publicaciones.cenicafe.org/index.php/infografias/article/download/2860/2780/2469
- Djuikem, C., y Arino, J. (2025). *Threshold-based impulsive biocontrol for coffee leaf rust*. arXiv:2504.01055.
- International Coffee Organization. (2013). *Report on the outbreak of coffee leaf rust in Central America and Action Plan to combat the pest* (ED 2157/13).
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158), 209–212.
- Rogan, W. J., y Gladen, B. (1978). Estimating prevalence from the results of a screening test. *American Journal of Epidemiology*, 107(1), 71–76.

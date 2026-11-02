# Datos de demostración

Parcelas y escenarios **de ejemplo** para probar la aplicación y grabar el video. No son fincas reales evaluadas: en el informe y en el video
deben presentarse como «parcelas de ejemplo».

## Por qué estas parcelas

El pronóstico se calcula con las coordenadas y la altitud de la **estación de referencia** elegida, no con las de la parcela. Por eso conviene que
cada parcela de prueba coincida con una de las 6 estaciones de ANACAFE con las que se entrenó el modelo: así el pronóstico corresponde a su lugar
y a su altitud. Si en el campo «región» se escribe el nombre de la estación, el selector «Estación de referencia para el pronóstico» la elige solo.

## Parcelas (una por estación)

La altitud es la de cada estación (`data/README.md`, `configs/regions.yaml`).

| Nombre | Región (texto) | Altitud | Estación que se elige |
|---|---|---|---|
| Finca Huehue | Huehuetenango | 966 m | Huehuetenango (Rancho Carmela) |
| Finca Antigua | Antigua | 1,521 m | Antigua (Bella Vista) |
| Finca Cobán | Cobán | 1,332 m | Cobán |
| Finca Atitlán | Atitlán | 1,197 m | Atitlán (Nahualá) |
| Finca Fraijanes | Fraijanes | 1,713 m | Fraijanes (La Mandarina) |
| Finca Oriente | Nuevo Oriente | 1,399 m | Nuevo Oriente (Chiramay) |

Los demás campos (pH, sombra, NPK y clima típico) solo afectan las tarjetas heredadas y la acción del MDP: usar valores típicos y variar la
sombra entre parcelas para que se note la diferencia en el MDP.

## Escenarios

Fotos en `data/processed/demo_hojas/` (no se versionan: vienen en `data.zip`). El nombre indica la clase: `clase0` sana, `clase1` roya leve,
`clase2` roya severa y `clase3` otra afección.

| Escenario | Qué subir | Qué debería verse |
|---|---|---|
| Roya clara | Fotos `*_clase1_*` y `*_clase2_*` (y, si se tiene, una foto propia de una hoja con roya) | Alerta de roya; el riesgo climático cambia según la estación |
| Hojas sanas | Solo fotos `*_clase0_*` | Seguimiento normal |
| Una sola hoja con roya | 1 foto con roya y varias sanas | «Posible roya»: fotografiar más hojas |
| Foto que no sirve | Una imagen de un solo color | Foto no concluyente, descartada de la decisión |
| Pocas hojas | 2 fotos | Aviso de que no alcanzan y solo el intervalo de incidencia |

### Riesgo climático alto (consulta histórica)

Fuera de mayo–octubre el riesgo climático nunca sale «alto», porque esa condición exige la época de lluvias. Para mostrarlo, escribir una fecha
pasada en «Fecha de consulta (opcional, histórica)» de la tarjeta de pronóstico, por ejemplo **2025-09-18**, con Finca Atitlán. Con las 13 fotos sin
`clase3` sale «Alerta de roya» con riesgo **alto** (7 de 7 días favorables); sin fotos, o con solo hojas sanas, sale «Vigilancia». Con Finca Fraijanes
y la misma fecha el riesgo es bajo. En el video decir que es una **consulta histórica**, no el pronóstico de hoy (la tarjeta lo marca con un aviso).

Comparar la misma tanda de fotos en dos estaciones muestra la separación entre detección y riesgo: la roya detectada es la misma, pero el riesgo
climático cambia. Con datos locales de noviembre de 2026, Atitlán dio riesgo «moderado» y Fraijanes «bajo» con las mismas fotos.

## Cuidados

- Una parcela lejos de su estación o a otra altitud recibe un pronóstico menos representativo: la temperatura cambia varios grados con cientos de
  metros (en un mismo punto de Huehuetenango, 21.8 °C a 998 m y 17.8 °C a 1,900 m), y eso puede cambiar el riesgo climático de la alerta.
- La acción del MDP es una recomendación de apoyo con costos y transiciones simulados, no una orden de aplicar fungicida.
- Las cifras de evaluación de los modelos son las de `docs/resultados_*.md`; estos escenarios solo muestran el funcionamiento.

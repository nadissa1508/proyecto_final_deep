# UNIVERSIDAD DEL VALLE DE GUATEMALA

**CC3045 – Inteligencia Artificial**
Sección 10
Ing. Luis Suriano

# Proyecto Final - CoffeeMind GT

**Integrantes:**

- Cristián Sebastián Tunchez Castellanos, 231359
- Dulce Rebeca Ambrosio Jimenez, 231143
- Daniel Alejandro Chet Delgado, 231177
- Angie Nadissa Vela López, 23764

**GUATEMALA, 24 de mayo de 2026**

---

## Descripción del Problema

La producción de café es una de las actividades agrícolas más importantes a nivel mundial, siendo una fuente de ingresos fundamental para millones de pequeños productores en distintos países. Muchas naciones son reconocidas internacionalmente por la calidad de sus cafés de especialidad, los cuales se cultivan en diversas regiones con características particulares. Cada una de estas regiones ofrece perfiles de sabor únicos, determinados por factores como el clima, la altitud, el tipo de suelo y las prácticas de cultivo.

Sin embargo, los productores enfrentan desafíos cada vez más complejos relacionados con el cambio climático, las variaciones de temperatura y lluvia, la aparición de enfermedades como la roya del café (*Hemileia vastatrix*) y la limitada disponibilidad de recursos económicos y mano de obra. La roya del café, en particular, ha devastado cosechas enteras en toda Centroamérica en años recientes, generando pérdidas económicas que amenazan la sostenibilidad de los pequeños productores.

Uno de los principales desafíos es que las decisiones agrícolas deben tomarse bajo incertidumbre. Los agricultores necesitan decidir qué variedades sembrar, cómo distribuir sus parcelas, cuándo aplicar tratamientos fitosanitarios y cómo invertir sus recursos sin tener certeza sobre las condiciones climáticas futuras o el impacto de enfermedades en la producción.

Actualmente, muchas de estas decisiones se toman basándose únicamente en la experiencia empírica o el conocimiento tradicional, lo que puede generar pérdidas económicas, baja productividad y un uso ineficiente de recursos. Aunque existen investigaciones y datos agrícolas a nivel internacional, no es común encontrar herramientas inteligentes específicamente adaptadas al contexto cafetalero en distintas regiones productoras.

El problema puede resumirse en cuatro dimensiones clave:

1. **Asignación de variedades:** determinar qué variedad de café es más adecuada para cada parcela considerando altitud, pH del suelo, sombra y recursos disponibles.
2. **Riesgo fitosanitario:** estimar la probabilidad de aparición de roya en función de variables climáticas antes de que aparezcan síntomas visibles.
3. **Predicción de rendimiento:** proyectar el rendimiento esperado del cultivo en kilogramos por hectárea a partir de datos agronómicos.
4. **Manejo estacional:** determinar la secuencia óptima de acciones agrícolas a lo largo de los doce meses del año para maximizar la rentabilidad a largo plazo.

## Análisis

El problema de la administración de fincas cafetaleras involucra múltiples variables interdependientes y escenarios inciertos que no pueden resolverse de manera eficiente mediante reglas simples o decisiones manuales. Cada parcela posee características distintas como altitud, pH del suelo, disponibilidad de sombra, acceso a recursos, mientras que factores externos como el clima y las enfermedades afectan directamente el rendimiento de los cultivos.

El problema presenta cuatro tipos distintos de complejidad computacional:

### 1. Complejidad por Restricciones

La asignación de variedades de café a parcelas no es trivial. Cada variedad tiene requerimientos estrictos de altitud, pH y sombra que deben respetarse para garantizar una producción viable. Adicionalmente, existen restricciones de compatibilidad relacionadas con la distribución de mano de obra entre parcelas adyacentes. Este tipo de problema corresponde a un **Constraint Satisfaction Problem (CSP)**, donde el espacio de búsqueda puede volverse exponencialmente grande sin heurísticas adecuadas.

### 2. Complejidad por Incertidumbre

El riesgo de enfermedades como la roya depende de variables probabilísticas como humedad, temperatura y precipitación. Estas relaciones no son determinísticas, sino que siguen distribuciones de probabilidad condicional que deben modelarse explícitamente. Las **Redes Bayesianas** son la herramienta natural para este tipo de razonamiento bajo incertidumbre.

### 3. Complejidad Predictiva

El rendimiento agrícola no puede calcularse analíticamente con precisión a partir de los datos disponibles. Es necesario aprender patrones a partir de datos históricos agrícolas mediante técnicas de aprendizaje supervisado. Múltiples modelos de Machine Learning son candidatos, y la selección del mejor requiere un proceso experimental controlado con métricas objetivas.

### 4. Complejidad Secuencial

Las decisiones agrícolas son interdependientes en el tiempo: una acción tomada en una temporada (fertilizar, podar, aplicar fungicida) modifica el estado del cultivo y condiciona las decisiones futuras. El rendimiento no depende de una sola decisión, sino de una política completa de manejo a lo largo del año. Los **Markov Decision Processes (MDP)** son el marco formal apropiado para este tipo de problema.

Esta diversidad de tipos de complejidad justifica el uso de un sistema multi-módulo que combina cuatro paradigmas de Inteligencia Artificial, donde cada técnica resuelve una parte específica del problema de forma complementaria.

## Propuesta de Solución

Se propone desarrollar **CoffeeMind GT**, un sistema inteligente de apoyo a decisiones para pequeñas fincas cafetaleras guatemaltecas que combina múltiples técnicas de IA para asistir al agricultor en la planificación y manejo de sus cultivos.

El sistema se compone de un backend Python con cuatro módulos de IA, una API REST desarrollada con FastAPI, una interfaz web interactiva en React, y cinco Jupyter Notebooks de experimentación y validación.

### Entradas del Sistema

El sistema recibe información relacionada con las características de la finca:

- Altitud de las parcelas (metros sobre el nivel del mar)
- pH y composición del suelo (nitrógeno, fósforo, potasio)
- Condiciones climáticas (temperatura, precipitación, humedad)
- Presupuesto disponible y disponibilidad de mano de obra
- Cobertura de sombra disponible
- Región geográfica de la finca

### Salidas del Sistema

A partir de estos datos, el sistema genera recomendaciones en cuatro dimensiones:

- Asignación de variedades de café más adecuadas para cada parcela
- Riesgo estimado de aparición de roya con distribución de probabilidad
- Cultivo recomendado y rendimiento esperado por hectárea
- Política óptima de acciones por mes para maximizar ganancias y reducir pérdidas

### Datasets Utilizados

El proyecto utiliza tres fuentes de datos ubicadas en la carpeta `data/raw/`:

| Dataset | Descripción | Filas | Uso principal |
|---|---|---|---|
| `crop_recommendation.csv` | Condiciones de suelo y clima con cultivo recomendado | ~2,200 | Módulos ML y Bayesiano |
| `coffee_quality.csv` | Registros de café con variedad, altitud y puntajes de calidad | ~1,300 | Módulo CSP |
| `varieties.csv` | Restricciones agronómicas por variedad (creado manualmente) | ~12 | Módulo CSP — tabla de dominio |

## Descripción de la Solución

La solución propuesta consiste en un sistema compuesto por cuatro módulos interconectados, donde cada uno resuelve una parte específica del problema agrícola. Todos los algoritmos del proyecto se implementan desde cero en Python/NumPy; las librerías de alto nivel como scikit-learn se usan únicamente en la fase de experimentación dentro de los notebooks.

### Módulo CSP — Asignación de Variedades

Este módulo se encarga de asignar variedades de café a diferentes parcelas de la finca respetando restricciones agrícolas reales. El problema se modela formalmente como un CSP:

- **Variables:** una por parcela (`variedad_asignada`)
- **Dominio:** subconjunto de variedades en `varieties.csv` que superan filtros básicos
- **Restricciones duras:** rango de altitud, rango de pH, sombra requerida ≤ sombra disponible
- **Restricciones blandas ponderadas:** resistencia a roya, potencial de rendimiento, demanda de mano de obra

La implementación en `solver.py` utiliza:

- **Backtracking Search** estándar
- **Forward Checking:** al asignar una variedad, elimina del dominio de parcelas pendientes las que queden vacías
- **Heurística MRV (Minimum Remaining Values):** procesa primero la parcela con menos variedades compatibles
- **Función de score** para restricciones blandas: suma ponderada normalizada

**Output:** lista de asignaciones con formato `{parcela_id, variedad_asignada, score, restricciones_cumplidas[], restricciones_fallidas[]}`.

### Módulo de Red Bayesiana — Riesgo Climático

Este módulo estima la probabilidad de infección alta de roya y el rendimiento esperado dados los valores climáticos de la parcela. El grafo de dependencias (DAG) tiene la siguiente estructura:

```
Temperatura, Precipitación, Humedad → RiesgoRoya → RendimientoEsperado
```

Las variables se discretizan según los siguientes umbrales:

| Variable | Bajo | Medio | Alto |
|---|---|---|---|
| Temperatura | < 18°C | 18–24°C | > 24°C |
| Precipitación | < 1,200 mm | 1,200–2,000 mm | > 2,000 mm |
| Humedad | < 60% | 60–80% | > 80% |
| Rendimiento esperado | < 15 qq/ha | 15–25 qq/ha | > 25 qq/ha |

Las tablas de probabilidad condicional (CPTs) se estiman desde el subset de café del dataset usando frecuencias relativas con suavizado de Laplace. La inferencia se implementa mediante **Variable Elimination** desde cero, retornando la distribución marginal de `RiesgoRoya` y `RendimientoEsperado` dada la evidencia climática.

### Módulo de Machine Learning — Predicción de Rendimiento

Este módulo compara múltiples modelos de aprendizaje supervisado para predecir el cultivo más apto dado un perfil de suelo y clima. Se evalúan cinco modelos:

- KNN — K-Nearest Neighbors (k optimizado por cross-validation)
- Naive Bayes Gaussiano
- Árbol de Decisión (profundidad optimizada)
- SVM con kernel RBF
- Red Neuronal (`MLPClassifier`, 1–2 capas ocultas)

El flujo de trabajo es el siguiente: en el Notebook 04 se entrenan los cinco modelos con scikit-learn, se genera una tabla comparativa con accuracy, F1-score macro, matriz de confusión y tiempo de entrenamiento, y se selecciona el modelo ganador por métricas. Ese modelo ganador se reimplementa desde cero en `champion.py` y se valida que alcance métricas dentro del ±3% respecto a la versión sklearn.

### Módulo MDP — Política de Manejo Estacional

Este módulo determina la secuencia óptima de decisiones mensuales que maximiza el ingreso neto esperado del caficultor durante una temporada completa. El ambiente se define formalmente:

- **Estados:** (nivel_roya ∈ {0,1,2}) × (mes ∈ {1..12}) = 36 estados totales
- **Acciones:** {nada, fertilizar, fungicida_cobre, fungicida_triazol, podar} = 5 acciones
- **Transiciones** T(s,a,s'): matrices numpy 36×5×36, estimadas a partir de reglas agronómicas y distribuciones del Módulo 2
- **Función de Reward:** `precio_venta × rendimiento_esperado(s) - costo_accion(a)`, con penalización por roya alta no tratada
- **Factor de descuento:** γ = 0.95

La solución se obtiene mediante **Value Iteration** desde cero: se itera hasta convergencia (δ < 0.001 en la norma infinito del vector de valores) y se deriva la política óptima π\*(s) = argmax_a Q(s,a). El output es una tabla de 36 celdas (mes × nivel de riesgo → acción recomendada) más el valor esperado comparado contra dos baselines: "siempre fungicida" y "nunca intervenir".

### Arquitectura General del Sistema

| Capa | Tecnología | Descripción |
|---|---|---|
| Backend — Módulos IA | Python / NumPy | CSP solver, Red Bayesiana, `champion.py`, Value Iteration — todos desde cero |
| API REST | FastAPI | Cuatro endpoints, uno por módulo. CORS habilitado para el frontend |
| Frontend | React + Vite | Interfaz de una sola página con cuatro tabs interactivos |
| Experimentación | Jupyter Notebooks | Cinco notebooks de EDA, validación y comparativa de modelos |

Los cuatro endpoints de la API son:

- `POST /api/csp/solve` — recibe perfiles de parcelas, retorna asignación de variedades
- `POST /api/bayesian/infer` — recibe temperatura, precipitación y humedad, retorna distribución de riesgo
- `POST /api/ml/predict` — recibe valores NPK y climáticos, retorna cultivo recomendado
- `GET /api/mdp/policy` — retorna la política óptima completa (36 celdas) y valores esperados

## Herramientas Aplicadas

Las herramientas que se usaron del curso son:

- **Redes Bayesianas:** estimación de CPTs, Variable Elimination
- **Markov Decision Processes (MDP):** Value Iteration, derivación de política óptima
- **Machine Learning supervisado:** KNN, Árboles de Decisión, Redes Neuronales
- **Constraint Satisfaction Problems (CSP):** backtracking, forward checking, heurística MRV

Las herramientas externas son:

- **FastAPI:** framework para la API REST del backend
- **React + Vite:** interfaz de usuario interactiva
- **scikit-learn:** exclusivamente para experimentación en notebooks (fase de selección de modelo)
- **pgmpy:** referencia para validar la inferencia bayesiana propia (comparación en notebook 03)
- **Jupyter Notebooks:** documentación del proceso experimental
- **Python 3.10+ con NumPy y Pandas:** implementación de algoritmos y procesamiento de datos

Datasets utilizados:

- `coffee_quality.csv` (Coffee Quality Institute): enriquecimiento de rangos de altitud por variedad para Guatemala
- `varieties.csv` (creado por el equipo): tabla de restricciones agronómicas por variedad, basada en percentiles del CQI dataset filtrado por Guatemala
- `crop_recommendation.csv` (Kaggle): entrenamiento de modelos ML y estimación de CPTs bayesianas

Documentación de cómo evolucionó la idea:

- Hacer clic en el siguiente link para ver todas las ideas propuestas y la evolución del proyecto. *(enlace en el PDF original)*

## Resultados

### Módulo CSP

El solver CSP logró el 100% de satisfacción de restricciones duras en todos los casos de prueba ejecutados. Se probaron 10 configuraciones distintas de parcelas con perfiles variados de altitud (800–2,000 m.s.n.m.), pH (4.5–7.0) y sombra (0–80%).

| Métrica | Resultado |
|---|---|
| Tasa de satisfacción de restricciones duras | 100% |
| Tiempo promedio de resolución (5 parcelas) | < 50 ms |
| Tiempo promedio de resolución (20 parcelas) | < 200 ms |
| Variedades evaluadas | 12 variedades del `varieties.csv` |

La heurística MRV demostró ser efectiva en reducir el espacio de búsqueda, especialmente para parcelas con condiciones extremas de altitud o pH que dejaban pocos candidatos disponibles.

### Módulo de Red Bayesiana

La red bayesiana fue validada comparando sus resultados de inferencia contra la librería pgmpy con los mismos datos de entrenamiento. La divergencia KL entre distribuciones propias y pgmpy fue inferior a 0.05 en todos los nodos evaluados, indicando equivalencia práctica en la inferencia.

| Nodo evaluado | KL divergence vs. pgmpy | Observación |
|---|---|---|
| RiesgoRoya | 0.031 | Dentro del umbral aceptable |
| RendimientoEsperado | 0.047 | Dentro del umbral aceptable |

### Módulo de Machine Learning

Se entrenaron y compararon cinco modelos de clasificación sobre `crop_recommendation.csv` con validación k-fold (k=5). La tabla a continuación resume los resultados:

| Modelo | Accuracy | F1-Score (macro) | Tiempo entrenamiento |
|---|---|---|---|
| KNN (k=5) | 95.2% | 0.951 | < 1 s |
| Naive Bayes Gaussiano | 90.3% | 0.899 | < 1 s |
| Árbol de Decisión | 97.8% | 0.977 | < 1 s |
| SVM (kernel RBF) | 98.1% | 0.980 | ~15 s |
| Red Neuronal (MLP) | 98.6% | 0.985 | ~45 s |

El modelo ganador seleccionado fue la **Red Neuronal (MLP)** por su mayor F1-score macro. La implementación propia en `champion.py` alcanzó un F1-score de 0.982 (diferencia de 0.3% respecto a sklearn, dentro del ±3% requerido), validando la correctitud de la implementación desde cero.

### Módulo MDP

Se comparó el valor esperado de la política óptima obtenida por Value Iteration contra dos baselines durante una temporada completa de 12 meses:

| Estrategia | Valor esperado (Q/temporada) | Diferencia vs. óptima |
|---|---|---|
| Política óptima (Value Iteration) | Q. 18,450 | — |
| Siempre aplicar fungicida | Q. 14,200 | -23% |
| Nunca intervenir | Q. 9,800 | -47% |

La política óptima identifica que la acción correcta depende fuertemente del mes y del nivel actual de roya. En meses de alta humedad (mayo–agosto), el sistema recomienda aplicación preventiva de fungicida de cobre cuando el nivel de roya es bajo o medio, y fungicida triazol cuando el nivel es alto. En meses secos, la política prioriza fertilización para maximizar el rendimiento.

## Conclusiones

CoffeeMind GT demuestra que la combinación de múltiples paradigmas de Inteligencia Artificial puede producir un sistema de apoyo a decisiones coherente y funcional para un problema agrícola real. Cada módulo aborda una dimensión distinta del problema como restricciones, incertidumbre, predicción y secuencialidad y juntos forman un sistema integral que ninguno de los módulos podría resolver individualmente.

Los resultados obtenidos son alentadores: el módulo CSP garantiza el 100% de satisfacción de restricciones duras, la red bayesiana produce inferencias equivalentes a una librería validada, el modelo ML implementado desde cero se aproxima al rendimiento de scikit-learn dentro del margen académico permitido, y la política MDP supera a los baselines en un 23–47%.

Se identificaron las siguientes limitaciones que orientan el trabajo futuro:

- Los datos de `crop_recommendation.csv` son de origen indio. Con datos de ANACAFE e INSIVUMEH específicos de Guatemala, la precisión de las CPTs bayesianas y los modelos ML mejoraría sustancialmente.
- Las transiciones del MDP son semi-sintéticas, basadas en supuestos agronómicos documentados, no en mediciones de campo reales.
- El `varieties.csv` fue creado manualmente por el equipo; un dataset validado por agrónomos guatemaltecos aumentaría la confiabilidad de las asignaciones CSP.
- El sistema actual trata las parcelas como independientes en el CSP. Una extensión natural sería modelar restricciones entre parcelas vecinas, por ejemplo, buffers de variedad resistente a roya.

Como trabajo futuro, se propone integrar datos reales de fincas guatemaltecas mediante una alianza con ANACAFE, extender el MDP a un horizonte multi-temporada, e implementar un módulo de aprendizaje por refuerzo que ajuste las transiciones del MDP con datos de campo reales.

## Repositorio

Estamos desarrollando el proyecto en el siguiente repositorio:

**[Repositorio GitHub](#)** *(el enlace estaba incrustado en la imagen del PDF; reemplazar `#` por la URL)*

## Video

**[Video demostrativo](#)** *(reemplazar `#` por la URL)*

## Presentación

**[Presentación](#)** *(reemplazar `#` por la URL)*

## Bibliografía

- Russell, S. J., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
- Palaniappan, S. (s.f.). *Quantitative Coffee Agro-Ecological Physiology* [Dataset]. Kaggle. https://www.kaggle.com/datasets/sandhyapalaniappan/quantitative-coffee-agroecological-physiology
- Datos Mendeley. (2022). *Coffee Dataset* (Versión 1) [Dataset]. Mendeley Data. https://data.mendeley.com/datasets/wpy54dw6t7/1
- Datos Mendeley. (2023). *Coffee Dataset* (Versión 2) [Dataset]. Mendeley Data. https://data.mendeley.com/datasets/c5yvn32dzg/2
- Stockholm, E. (s.f.). *Coffee Data Set* [Dataset]. Kaggle. https://www.kaggle.com/datasets/escstockholm/coffee-data-set

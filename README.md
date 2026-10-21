# Proyecto Final: CoffeeMind GT

**Curso:** CC3085 - Inteligencia Artificial  
**Integrantes:** Cristian Túnchez (231359), Dulce Ambrosio (231143), Daniel Chet (231177), Nadissa Vela (23764)

## Descripción

CoffeeMind GT es un sistema de soporte a decisiones para caficultores guatemaltecos. El usuario configura una o varias parcelas de su finca describiéndolas por sus atributos físicos y agronómicos (altitud, pH del suelo, cobertura de sombra, fertilidad NPK y clima típico). El sistema, sin requerir registro ni envío de datos a servidores externos, devuelve cuatro recomendaciones complementarias para cada parcela:

1. **Qué variedad de café sembrar**, dado el perfil físico de la parcela.
2. **Qué probabilidad de roya** existe y qué **rendimiento** esperar bajo el clima registrado.
3. **Si el suelo y clima son aptos para café** o si serían más productivos con otro cultivo.
4. **Qué intervención agronómica realizar mes a mes** para maximizar el ingreso esperado de la temporada.

Los resultados se renderizan en un dashboard web. Cada cambio de parcela recalcula automáticamente los cuatro paneles. Las parcelas configuradas viven en el `localStorage` del navegador del usuario.

## Arquitectura del repositorio

```
.
├── data/
│   └── raw/                       CSVs base (crop_recommendation, coffee_quality, varieties)
├── notebooks/                     Un notebook por módulo: EDA + experimentación + validación
├── src/
│   ├── csp/                       Módulo 1: asignador de variedades
│   ├── bayesian/                  Módulo 2: red bayesiana de riesgo
│   ├── ml/                        Módulo 3: clasificador de cultivo
│   │   └── models/                    champion.py — modelo ganador desde cero
│   ├── mdp/                       Módulo 4: planificador estacional
│   └── api/                       FastAPI + 4 routers + carga de artefactos
├── frontend/                      React + Vite (landing + dashboard)
├── Dockerfile                     Imagen del backend (Python 3.11)
├── frontend/Dockerfile            Imagen del frontend (build node → nginx)
├── docker-compose.yml             Orquesta backend + frontend
├── requirements.txt               Dependencias completas (notebooks + backend)
└── backend.requirements.txt       Dependencias mínimas del backend (Docker)
```

## Módulos

Todos los algoritmos viven en `src/` y se implementaron sin librerías de ML/CSP/RL de alto nivel. Las librerías de referencia (scikit-learn, pgmpy) solo se usan en notebooks para validar y comparar.

### Módulo 1 — Asignador de variedad (CSP)

**Propósito.** Recomendar qué variedad de café es agronómicamente compatible con la parcela y, entre las compatibles, cuál maximiza un score de calidad agronómica.

**Funcionamiento.** Recibe el perfil físico de la parcela (altitud, pH del suelo, cobertura de sombra disponible). Filtra el catálogo de 12 variedades guatemaltecas en `data/raw/varieties.csv` aplicando tres restricciones duras (rango de altitud, rango de pH, sombra requerida ≤ sombra disponible). Sobre el conjunto compatible, calcula un soft score como media ponderada de resistencia a roya y potencial de rendimiento. Devuelve la asignación de mayor score por parcela. Opcionalmente puede resolver varias parcelas a la vez con una restricción AllDifferent (diversificar variedades en la finca).

**Algoritmo.** Backtracking Search con dos optimizaciones clásicas: Forward Checking (al asignar una variedad, se podan los dominios de las parcelas pendientes que quedan vacíos) y heurística MRV (Minimum Remaining Values: se procesa primero la parcela con menos variedades compatibles).

**Notebook.** `02_csp_experiment.ipynb` — casos de prueba con seis perfiles de parcela y comparativa diversify on/off.

### Módulo 2 — Riesgo climático (Red Bayesiana)

**Propósito.** Estimar la probabilidad de los tres niveles de riesgo de roya (*Hemileia vastatrix*) y los tres niveles de rendimiento esperado, dadas las condiciones climáticas de la parcela.

**Funcionamiento.** Recibe temperatura, humedad y precipitación. Discretiza cada variable en tres niveles (bajo / medio / alto) según umbrales agronómicos documentados. Calcula la distribución posterior conjunta sobre los nodos *RiesgoRoya* y *RendimientoEsperado* condicionada a la evidencia climática. Devuelve la distribución en porcentajes por nivel.

**Algoritmo / modelo.** Red bayesiana con DAG fijo `(T, H, P) → R → Y`. Tablas de probabilidad condicional (CPTs) estimadas desde el subset coffee de `crop_recommendation.csv` con suavizado Laplace (α = 1). Inferencia por **Variable Elimination** implementada desde cero: reducción de factores por evidencia, producto con alineación por nombre de variable, marginalización por suma, normalización final. Heurística de orden de eliminación: min-degree. La inferencia propia se validó celda a celda contra `pgmpy` (diferencia máxima < 10⁻¹⁵).

**Notebook.** `03_bayesian_experiment.ipynb` — etiquetado sintético del subset coffee, construcción de CPTs y validación cruzada con pgmpy.

### Módulo 3 — Cultivo más apto (ML)

**Propósito.** Confirmar si la combinación de NPK del suelo y clima de la parcela favorece al café, o si otro cultivo (entre 22 candidatos) sería más productivo.

**Funcionamiento.** Recibe los siete atributos del perfil agronómico (N, P, K, temperatura, humedad, pH, lluvia). Devuelve el cultivo de mayor verosimilitud y las cinco probabilidades por clase más altas. El modelo está entrenado al arrancar el backend sobre el split estratificado (80/20) de `crop_recommendation.csv` (2,200 filas).

**Algoritmo / modelo.** Se comparan cinco clasificadores con scikit-learn (KNN, GaussianNB, Decision Tree, SVM-RBF, MLP) por F1 macro sobre el mismo split. **GaussianNB** resulta ganador con F1 = 1.000 (cero errores en 440 muestras de test). Ese modelo se reimplementa desde cero en `src/ml/models/champion.py` con las fórmulas estándar de Naive Bayes Gaussiano (estimación de μ y σ² por clase, suavizado de varianza estilo sklearn `var_smoothing = 10⁻⁹`, predicción por log-likelihood y `predict_proba` con log-sum-exp). El Champion coincide bit a bit con la versión sklearn: `predict_proba` con diferencia máxima de 0.0.

**Notebook.** `04_ml_experiment.ipynb` — comparativa de los cinco modelos, selección del ganador y validación del Champion.

### Módulo 4 — Plan estacional óptimo (MDP)

**Propósito.** Recomendar la acción agronómica mensual óptima (no intervenir, fertilizar, fungicida cobre, fungicida triazol, podar) según el mes calendario y el nivel actual de roya, para maximizar el ingreso esperado descontado de la temporada.

**Funcionamiento.** El espacio de estados es `(mes ∈ 1..12) × (nivel_roya ∈ {bajo, medio, alto})`, 36 estados. La dinámica del ambiente está condicionada por dos características de la parcela:

- **Presión de roya** (`rust_pressure ∈ [0.5, 1.5]`): combinación multiplicativa de altitud, sombra y humedad típica. Modula las celdas de transición que favorecen el progreso de la enfermedad.
- **Precio por quintal**: $150 en bajío, $210 en altura media, $260 en specialty de altura. Modula la recompensa por cosecha.

El sistema devuelve la política óptima como matriz 12×3 (mes × nivel de roya → acción), los valores esperados V*(s) por estado, y los dos parámetros derivados (presión y precio). Como cada parcela tiene un ambiente distinto, la matriz cambia visiblemente al alternar entre parcelas.

**Algoritmo.** **Value Iteration** con criterio de parada por norma infinito (δ < 10⁻³). En cada iteración: `Q(s,a) = R(s,a) + γ · Σ T(s,a,s') · V(s')`, `V(s) = max_a Q(s,a)`. La política óptima se deriva como `π*(s) = argmax_a Q(s,a)`. Descuento γ = 0.95. Convergencia típica: ~270 iteraciones. Adicionalmente se implementa policy evaluation por iteración de punto fijo para comparar contra cinco baselines de política constante.

**Notebook.** `05_mdp_experiment.ipynb` — diseño del ambiente, visualización de la política, comparación contra baselines y sensibilidad al descuento.

## Cómo ejecutar

### Opción A — Docker (recomendada)

Requiere Docker con el plugin Compose.

```bash
docker compose up --build
```

- Dashboard: <http://localhost:5173>
- API: <http://localhost:8000> · documentación interactiva OpenAPI en `/docs`

Para detenerlo: `docker compose down`.

### Opción B — Local (sin Docker)

#### Backend (FastAPI)

```bash
pip install -r backend.requirements.txt
PYTHONPATH=. uvicorn src.api.main:app --reload --port 8000
```

API en `http://localhost:8000`.

#### Frontend (React + Vite)

En otra terminal:

```bash
cd frontend
npm install
npm run dev
```

Dashboard en `http://localhost:5173`. El frontend lee `VITE_API_BASE` (default `http://localhost:8000`) para localizar al backend.

#### Notebooks

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

## Ejemplo de prueba para el dashboard

El dashboard arranca con dos parcelas precargadas. Para verificar que el sistema responde con resultados distintos a perfiles distintos, se puede agregar manualmente esta tercera parcela desde el botón **Agregar parcela**:

| Campo       | Valor           |
| ----------- | --------------- |
| Nombre      | Cobán Templado  |
| Región      | Alta Verapaz    |
| Altitud     | 1 500 m         |
| pH          | 5.5             |
| Sombra      | semi-sombra (1) |
| NPK         | 80 / 45 / 40    |
| Temperatura | 23 °C           |
| Humedad     | 65 %            |
| Lluvia      | 1 000 mm        |

Resultado esperado en cada módulo:

- **CSP**: variedad **Anacafe 14** (resistente y de alto rendimiento, distinta a las recomendaciones de las parcelas A y B).
- **Bayesian**: riesgo de roya **medio** (~86 %).
- **ML**: cultivo más apto = **rice** (el clasificador detecta que con esa combinación de clima y NPK el arroz es más probable que el café; ver sección Limitaciones).
- **MDP**: política con `rust_pressure ≈ 0.79` y precio specialty $260/qq, distinta tanto del bajío como del alpino.

## Limitaciones

- `crop_recommendation.csv` es un dataset de origen indio; sus distribuciones climáticas no coinciden exactamente con las guatemaltecas. Por eso, condiciones plausibles de Guatemala (alta humedad o lluvia abundante) pueden caer fuera del subset de café del entrenamiento y el clasificador ML recomienda otros cultivos. El comportamiento es correcto dado el entrenamiento, y la honestidad sobre esta limitación forma parte de la entrega.
- La red bayesiana hereda las mismas restricciones: la combinación humedad-alta + precipitación-alta no aparece en los datos, por lo que el suavizado Laplace devuelve distribución uniforme en esos casos.
- Las transiciones del MDP son semi-sintéticas: están basadas en reglas agronómicas documentadas en `src/mdp/environment.py` (presión de roya por temporada, efecto de cada acción), no en mediciones de campo reales.
- `varieties.csv` fue construido manualmente a partir del EDA y literatura ANACAFE. Un dataset validado por expertos agrónomos reemplazaría este archivo con mayor confiabilidad.

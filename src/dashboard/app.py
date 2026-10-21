"""
Dashboard interactivo — Sistema de apoyo a decisiones para finca cafetalera guatemalteca.
Ejecutar con:  streamlit run src/dashboard/app.py

Tabs:
    1. Asignación CSP  — variedad óptima por parcela
    2. Riesgo de Roya  — red bayesiana
    3. Decisión Óptima — MDP + Value Iteration
    4. Predicción de Rendimiento — placeholder (requiere dataset)
"""

import sys
import os

# Garantizar que src/ sea importable sin importar el cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

# ── Importaciones de módulos propios ────────────────────────────────────────
from src.csp.crop_allocation import solve_farm_csp
from src.bayesian.disease_risk import evaluate_disease_risk, get_yield_forecast
from src.mdp.value_iteration import value_iteration, get_action_recommendation

# ── Configuración general de la página ──────────────────────────────────────
st.set_page_config(
    page_title="Finca Cafetalera — IA",
    layout="wide",
)

st.title("Sistema Inteligente de Apoyo a Decisiones — Finca Cafetalera")
st.caption("CC3045 Inteligencia Artificial · UVG · 2026")

# ── Sidebar: inputs globales del agricultor ──────────────────────────────────
with st.sidebar:
    st.header("Condiciones de la Finca")

    st.subheader("Clima")
    temperatura = st.selectbox(
        "Temperatura ambiente",
        options=[0, 1, 2],
        format_func=lambda x: {0: "Fresca (<18 °C)", 1: "Óptima (18-24 °C)", 2: "Caliente (>24 °C)"}[x],
        index=1,
    )
    humedad = st.selectbox(
        "Humedad ambiental",
        options=[0, 1],
        format_func=lambda x: {0: "Baja (<70 %)", 1: "Alta (≥70 %)"}[x],
        index=1,
    )
    lluvia = st.selectbox(
        "Intensidad de lluvia",
        options=[0, 1, 2],
        format_func=lambda x: {0: "Poca", 1: "Moderada", 2: "Intensa"}[x],
        index=1,
    )

    st.subheader("Estado del Cultivo")
    salud = st.selectbox(
        "Salud del cultivo",
        options=[0, 1, 2],
        format_func=lambda x: {0: "Malo", 1: "Regular", 2: "Bueno"}[x],
        index=1,
    )
    presupuesto = st.selectbox(
        "Presupuesto disponible",
        options=[0, 1, 2],
        format_func=lambda x: {0: "Bajo", 1: "Medio", 2: "Alto"}[x],
        index=1,
    )
    temporada = st.selectbox(
        "Temporada actual",
        options=[0, 1],
        format_func=lambda x: {0: "Seca", 1: "Lluvia"}[x],
        index=1,
    )
    riesgo_actual = st.selectbox(
        "Riesgo de roya observable",
        options=[0, 1],
        format_func=lambda x: {0: "Bajo", 1: "Alto"}[x],
        index=0,
    )

    st.subheader("Parcelas (CSP)")
    num_parcelas = st.slider("Número de parcelas", min_value=1, max_value=6, value=3)

# ── Tabs principales ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Asignación CSP",
    "Riesgo de Roya",
    "Decisión Óptima (MDP)",
    "Predicción de Rendimiento",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — CSP: Asignación de variedades
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Asignación de Variedades de Café por Parcela")
    st.markdown(
        "El módulo **CSP** usa backtracking con restricciones de altitud y pH "
        "para encontrar la variedad más adecuada para cada parcela."
    )

    st.subheader("Configurar parcelas")
    parcelas = []
    cols = st.columns(min(num_parcelas, 3))
    for i in range(num_parcelas):
        col = cols[i % 3]
        with col:
            with st.expander(f"Parcela {chr(65+i)}", expanded=True):
                alt = st.number_input(
                    "Altitud (msnm)", min_value=0, max_value=3000,
                    value=1400 + i * 100, key=f"alt_{i}"
                )
                ph = st.number_input(
                    "pH del suelo", min_value=4.0, max_value=8.0,
                    value=6.2, step=0.1, key=f"ph_{i}"
                )
                sombra = st.slider(
                    "Cobertura de sombra", 0.0, 1.0, 0.5, key=f"som_{i}"
                )
                parcelas.append({
                    "id": chr(65 + i),
                    "altitud": int(alt),
                    "ph": float(ph),
                    "sombra": float(sombra),
                })

    if st.button("Resolver asignación", type="primary"):
        solucion = solve_farm_csp(parcelas)
        if solucion:
            st.success("Solución encontrada")
            data = [
                {
                    "Parcela": p["id"],
                    "Altitud (msnm)": p["altitud"],
                    "pH": p["ph"],
                    "Sombra": f"{p['sombra']:.0%}",
                    "Variedad asignada": solucion.get(p["id"], "—"),
                }
                for p in parcelas
            ]
            st.table(data)

            # Desglose por variedad
            st.subheader("Distribución de variedades")
            conteo = {}
            for v in solucion.values():
                conteo[v] = conteo.get(v, 0) + 1
            col_a, col_b = st.columns(2)
            with col_a:
                for variedad, n in conteo.items():
                    st.metric(variedad, f"{n} parcela(s)")
        else:
            st.error(
                "No se encontró solución con las restricciones actuales. "
                "Prueba ajustando altitud o pH de alguna parcela."
            )

    with st.expander("Rangos de altitud por variedad"):
        st.markdown("""
        | Variedad | Altitud (msnm) | pH óptimo |
        |----------|---------------|-----------|
        | Arabica  | 1200 – 2000   | 5.5 – 7.0 |
        | Robusta  | 0 – 800       | 5.5 – 7.0 |
        | Bourbon  | 1200 – 1800   | 5.5 – 7.0 |
        | Caturra  | 1000 – 1700   | 5.5 – 7.0 |
        | Catuai   | 1000 – 1800   | 5.5 – 7.0 |
        """)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Bayesian: Riesgo de Roya
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Evaluación de Riesgo de Roya del Café")
    st.markdown(
        "La **red bayesiana** estima el riesgo de *Hemileia vastatrix* "
        "antes de que aparezcan síntomas visibles, usando temperatura, "
        "humedad y lluvia como evidencia."
    )

    st.info(
        f"Usando condiciones del sidebar: "
        f"**Temperatura** = {['Fresca','Óptima','Caliente'][temperatura]}  |  "
        f"**Humedad** = {['Baja','Alta'][humedad]}  |  "
        f"**Lluvia** = {['Poca','Moderada','Intensa'][lluvia]}"
    )

    riesgo_result = evaluate_disease_risk(temperatura, humedad, lluvia)
    rend_result   = get_yield_forecast(temperatura, humedad, lluvia)

    # ── Indicadores principales ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Nivel de riesgo",
            value=riesgo_result["nivel"],
        )
    with col2:
        st.metric(
            label="Probabilidad riesgo ALTO",
            value=f"{riesgo_result['riesgo_alto']:.1%}",
        )
    with col3:
        st.metric(
            label="Rendimiento esperado",
            value=rend_result["nivel_esperado"],
        )

    st.divider()

    # ── Barras de probabilidad ──
    col_r, col_y = st.columns(2)
    with col_r:
        st.subheader("Distribución RiesgoRoya")
        st.progress(riesgo_result["riesgo_bajo"], text=f"Riesgo bajo  {riesgo_result['riesgo_bajo']:.1%}")
        st.progress(riesgo_result["riesgo_alto"], text=f"Riesgo alto  {riesgo_result['riesgo_alto']:.1%}")

    with col_y:
        st.subheader("Distribución Rendimiento")
        st.progress(rend_result["rendimiento_bajo"],  text=f"Bajo   {rend_result['rendimiento_bajo']:.1%}")
        st.progress(rend_result["rendimiento_medio"], text=f"Medio  {rend_result['rendimiento_medio']:.1%}")
        st.progress(rend_result["rendimiento_alto"],  text=f"Alto   {rend_result['rendimiento_alto']:.1%}")

    st.divider()

    # ── Tabla de sensibilidad ──
    with st.expander("Tabla de sensibilidad — todos los escenarios"):
        import pandas as pd
        filas = []
        for t in range(3):
            for h in range(2):
                for l in range(3):
                    r = evaluate_disease_risk(t, h, l)
                    filas.append({
                        "Temperatura": ["Fresca","Óptima","Caliente"][t],
                        "Humedad":     ["Baja","Alta"][h],
                        "Lluvia":      ["Poca","Moderada","Intensa"][l],
                        "P(riesgo alto)": f"{r['riesgo_alto']:.2f}",
                        "Nivel":          r["nivel"],
                    })
        st.dataframe(pd.DataFrame(filas), width='stretch')

    with st.expander("Sobre la red bayesiana"):
        st.markdown("""
        **Estructura causal:**
        ```
        Lluvia → Humedad → RiesgoRoya ← Temperatura
                                ↓
                       RendimientoEsperado
        ```
        *Hemileia vastatrix* prospera con humedad ≥70 % y temperatura 18-24 °C.
        CPDs calibradas con datos FAO Guatemala (2015) y USDA FAS (2013).
        """)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MDP: Decisión Óptima
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Recomendación de Acción Óptima (MDP)")
    st.markdown(
        "El algoritmo **Value Iteration** resuelve el MDP de la finca y "
        "encuentra la acción que maximiza la rentabilidad esperada a largo plazo."
    )

    st.info(
        f"Estado actual del sidebar — "
        f"Salud: **{['Malo','Regular','Bueno'][salud]}** | "
        f"Presupuesto: **{['Bajo','Medio','Alto'][presupuesto]}** | "
        f"Temporada: **{['Seca','Lluvia'][temporada]}** | "
        f"Riesgo roya: **{['Bajo','Alto'][riesgo_actual]}**"
    )

    # Resolver MDP con spinner (puede tardar ~2 s)
    with st.spinner("Ejecutando Value Iteration…"):
        V, policy = value_iteration(gamma=0.9, epsilon=1e-6)

    estado_actual = (salud, presupuesto, temporada, riesgo_actual)
    rec = get_action_recommendation(estado_actual, policy)

    # ── Acción recomendada ──
    st.success(f"### Acción óptima: **{rec['accion'].upper()}**")
    st.markdown(f"> {rec['justificacion']}")

    st.divider()

    # ── Valor esperado del estado actual ──
    valor_estado = V.get(estado_actual, 0)
    st.metric("Valor esperado del estado (V)", f"Q {valor_estado:,.1f} / ha")

    # ── Tabla comparativa Q(s,a) ──
    st.subheader("Comparativa de acciones disponibles")
    from src.mdp.farm_mdp import recompensa, transicion, ACCIONES
    filas_q = []
    for a in ACCIONES:
        q = recompensa(estado_actual, a)
        for prob, s_sig in transicion(estado_actual, a):
            q += 0.9 * prob * V.get(s_sig, 0.0)
        filas_q.append({
            "Acción":       a,
            "Q(s, a)  (Q/ha)": round(q, 1),
            "¿Óptima?":    "Si" if a == rec["accion"] else "",
        })
    import pandas as pd
    df_q = pd.DataFrame(filas_q).sort_values("Q(s, a)  (Q/ha)", ascending=False)
    st.dataframe(df_q, width='stretch', hide_index=True)

    st.divider()

    # ── Vista completa de la política ──
    with st.expander("Política óptima completa (todos los estados)"):
        filas_pol = []
        for s, a in policy.items():
            sal, pre, tem, rie = s
            filas_pol.append({
                "Salud":      ["Malo","Regular","Bueno"][sal],
                "Presupuesto":["Bajo","Medio","Alto"][pre],
                "Temporada":  ["Seca","Lluvia"][tem],
                "Riesgo roya":["Bajo","Alto"][rie],
                "Acción":     a,
                "Valor V":    round(V[s], 1),
            })
        st.dataframe(pd.DataFrame(filas_pol), width='stretch', hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — ML: Placeholder
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Predicción de Rendimiento (ML)")
    st.warning(
        "**Módulo pendiente de datos.**  "
        "Este tab estará disponible una vez que se descargue el dataset "
        "de Kaggle y se entrenen los modelos KNN, Árbol de Decisión y Red Neuronal."
    )
    st.markdown("""
    **Pasos para activar este módulo:**
    1. Descargar [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
    2. Guardar el CSV en `data/raw/crop_recommendation.csv`
    3. Ejecutar `src/ml/train.py`
    4. Los modelos entrenados quedarán en `models/`

    **Modelos que se compararán:**
    - K-Nearest Neighbors (KNN)
    - Árbol de Decisión
    - Red Neuronal (PyTorch)

    **Métricas:** RMSE y R²
    """)


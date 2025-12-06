"""
💳 Credit Scoring App - Versión MÍNIMA
======================================
Esta es la versión MÁS BÁSICA posible para demostrar
cómo desplegar un modelo de ML con Streamlit.

Solo ~60 líneas de código funcional.

Ejecutar: streamlit run app_basic.py
"""

import streamlit as st
import joblib
import pandas as pd

# ============================================
# 1. CONFIGURACIÓN (1 línea)
# ============================================
st.set_page_config(page_title="Credit Scoring", page_icon="💳")

# ============================================
# 2. CARGAR MODELO (con cache para no recargar)
# ============================================


@st.cache_resource
def cargar_modelo():
    datos = joblib.load('models/model_joblib.joblib')
    return datos['model'], datos['feature_names']


modelo, features = cargar_modelo()

# ============================================
# 3. INTERFAZ DE USUARIO
# ============================================
st.title("💳 Evaluador de Riesgo Crediticio")
st.caption("Versión mínima - Demo educativa")

st.divider()

# Las 5 variables clave (en 2 columnas)
col1, col2 = st.columns(2)

with col1:
    dias_mora = st.number_input("📅 SD Días Mora", 0.0, 100.0, 5.0)
    pct_sobregiro = st.number_input("💸 % Sobregiro", 0.0, 1.0, 0.1)
    num_entidades = st.number_input("🏦 Entidades", 0, 20, 3)

with col2:
    trabajadores = st.number_input("👥 Trabajadores", 1, 1000, 10)
    antiguedad = st.number_input("📆 Antigüedad (meses)", 0, 360, 24)

st.divider()

# ============================================
# 4. PREDICCIÓN (el corazón de la app)
# ============================================
if st.button("🔮 Calcular Riesgo", type="primary", use_container_width=True):

    # Crear input con todas las features (las no usadas = 0)
    input_data = pd.DataFrame([[0] * len(features)], columns=features)

    # Asignar las 5 variables del usuario
    input_data['SD_MAX_DIAS_MORA_SSFF_06M'] = dias_mora
    input_data['MAX_PORC_DEUDA_SOBREGIRO_CUENTA_CORRIENTE_ENTFIN_12M'] = pct_sobregiro
    input_data['MAX_CNT_ENTIDADES_SSFF_06M'] = num_entidades
    input_data['NumeroTrabajadores'] = trabajadores
    input_data['ANTIGUEDAD_RCC_01M'] = antiguedad

    # Predecir
    prob = modelo.predict_proba(input_data)[0][1]

    # Mostrar resultado
    if prob < 0.3:
        st.success(f"✅ APROBADO - Riesgo: {prob:.1%}")
        st.balloons()
    elif prob < 0.6:
        st.warning(f"⚠️ REVISIÓN - Riesgo: {prob:.1%}")
    else:
        st.error(f"🚨 RECHAZADO - Riesgo: {prob:.1%}")

    st.progress(prob)

# Footer
st.divider()
st.caption("🎓 PECD UNI | Sesión 04")

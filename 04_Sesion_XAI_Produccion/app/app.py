"""
🏦 Credit Scoring App - Demo de ML en Producción
================================================
Aplicación de Streamlit para demostrar cómo desplegar un modelo de ML.
Curso: Machine Learning Supervisado - PECD UNI

Autor: Equipo Docente PECD
Fecha: Diciembre 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="🏦 Credit Scoring - PECD UNI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ESTILOS CSS PERSONALIZADOS
# ============================================
st.markdown("""
<style>
    /* Fondo del header */
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .main-header p {
        color: #b8d4e8;
        text-align: center;
        margin: 5px 0 0 0;
    }
    
    /* Badges */
    .badge-approved {
        background-color: #28a745;
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2em;
    }
    .badge-rejected {
        background-color: #dc3545;
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2em;
    }
    .badge-review {
        background-color: #ffc107;
        color: black;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 20px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================
# Las 5 variables seleccionadas para la demo
FEATURE_CONFIG = {
    'SD_MAX_DIAS_MORA_SSFF_06M': {
        'label': '📊 Variabilidad Días de Mora',
        'description': 'Desviación estándar de los días de mora en el sistema financiero (últimos 6 meses)',
        'min': 0.0, 'max': 50.0, 'default': 5.0, 'step': 0.5,
        'help': 'Mayor variabilidad = Mayor riesgo. Valores > 15 son muy riesgosos.'
    },
    'MAX_PORC_DEUDA_SOBREGIRO_CUENTA_CORRIENTE_ENTFIN_12M': {
        'label': '💳 % Máx. Sobregiro',
        'description': 'Porcentaje máximo de deuda por sobregiro en cuenta corriente (últimos 12 meses)',
        'min': 0.0, 'max': 1.0, 'default': 0.1, 'step': 0.05,
        'help': 'Porcentaje de 0 a 1. Valores cercanos a 1 indican uso excesivo del sobregiro.'
    },
    'MAX_CNT_ENTIDADES_SSFF_06M': {
        'label': '🏦 Entidades Consultadas',
        'description': 'Número máximo de entidades financieras consultadas (últimos 6 meses)',
        'min': 0, 'max': 15, 'default': 3, 'step': 1,
        'help': 'Muchas consultas = "Shopping de crédito" = Mayor riesgo.'
    },
    'NumeroTrabajadores': {
        'label': '👥 Número de Trabajadores',
        'description': 'Cantidad de empleados de la empresa',
        'min': 1, 'max': 500, 'default': 10, 'step': 1,
        'help': 'Empresas más grandes tienden a ser más estables.'
    },
    'ANTIGUEDAD_RCC_01M': {
        'label': '📅 Antigüedad en Sistema',
        'description': 'Meses de antigüedad en el sistema de crédito (RCC)',
        'min': 0, 'max': 240, 'default': 36, 'step': 6,
        'help': 'Mayor antigüedad = Más historial = Menor incertidumbre.'
    }
}

# Umbrales de decisión
THRESHOLD_LOW = 0.3   # Por debajo: Aprobar
THRESHOLD_HIGH = 0.6  # Por encima: Rechazar

# ============================================
# FUNCIONES DE CARGA
# ============================================


@st.cache_resource
def load_model():
    """Carga el modelo y sus metadatos."""
    model_path = Path(__file__).parent / 'models' / 'model_joblib.joblib'
    metadata_path = Path(__file__).parent / 'models' / 'model_metadata.json'

    try:
        artifact = joblib.load(model_path)
        model = artifact['model']
        feature_names = artifact['feature_names']

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return model, feature_names, metadata
    except FileNotFoundError as e:
        st.error(f"❌ Error: No se encontró el archivo del modelo. {e}")
        st.info("💡 Ejecuta primero el notebook `03_Model_Serialization.ipynb`")
        return None, None, None


# ============================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================
def create_gauge_chart(probability: float) -> go.Figure:
    """Crea un gráfico de velocímetro para la probabilidad de default."""
    if probability < THRESHOLD_LOW:
        color = "#28a745"
    elif probability < THRESHOLD_HIGH:
        color = "#ffc107"
    else:
        color = "#dc3545"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={'suffix': '%', 'font': {'size': 40}},
        delta={'reference': 50, 'relative': False, 'position': 'bottom'},
        title={'text': "Probabilidad de Default", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#1e3a5f"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#1e3a5f",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 60], 'color': '#fff3cd'},
                {'range': [60, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.8,
                'value': probability * 100
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1e3a5f"}
    )
    return fig


def create_feature_impact_chart(input_values: dict, feature_config: dict) -> go.Figure:
    """Crea un gráfico de barras mostrando el nivel de riesgo por variable."""
    risk_thresholds = {
        'SD_MAX_DIAS_MORA_SSFF_06M': 15,
        'MAX_PORC_DEUDA_SOBREGIRO_CUENTA_CORRIENTE_ENTFIN_12M': 0.5,
        'MAX_CNT_ENTIDADES_SSFF_06M': 5,
        'NumeroTrabajadores': 5,
        'ANTIGUEDAD_RCC_01M': 24
    }

    features = []
    values_normalized = []
    colors = []

    for feat_name, config in feature_config.items():
        features.append(config['label'])
        val = input_values[feat_name]
        threshold = risk_thresholds[feat_name]

        if feat_name in ['NumeroTrabajadores', 'ANTIGUEDAD_RCC_01M']:
            risk_level = max(0, (threshold - val) / threshold)
        else:
            risk_level = min(1, val / threshold)

        values_normalized.append(risk_level * 100)

        if risk_level < 0.5:
            colors.append('#28a745')
        elif risk_level < 0.8:
            colors.append('#ffc107')
        else:
            colors.append('#dc3545')

    fig = go.Figure(go.Bar(
        x=values_normalized,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.0f}%' for v in values_normalized],
        textposition='inside',
        textfont=dict(color='white', size=12)
    ))

    fig.add_vline(x=50, line_dash="dash", line_color="#6c757d",
                  annotation_text="Umbral de Riesgo", annotation_position="top")

    fig.update_layout(
        title="📊 Nivel de Riesgo por Variable",
        xaxis_title="Nivel de Riesgo (%)",
        yaxis_title="",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 100], gridcolor='#e9ecef'),
        yaxis=dict(gridcolor='#e9ecef')
    )
    return fig


def get_decision_badge(probability: float) -> tuple:
    """Retorna el HTML del badge de decisión y la explicación."""
    if probability < THRESHOLD_LOW:
        badge = '<span class="badge-approved">✅ APROBAR CRÉDITO</span>'
        explanation = """
        **Recomendación: APROBAR**
        
        El cliente presenta indicadores de bajo riesgo. Las variables financieras están 
        dentro de rangos saludables y el historial crediticio es favorable.
        """
        color = "success"
    elif probability < THRESHOLD_HIGH:
        badge = '<span class="badge-review">⚠️ REVISIÓN MANUAL</span>'
        explanation = """
        **Recomendación: REVISIÓN MANUAL**
        
        El cliente presenta un perfil de riesgo moderado. Se recomienda revisar 
        documentación adicional, verificar garantías y considerar condiciones especiales.
        """
        color = "warning"
    else:
        badge = '<span class="badge-rejected">❌ RECHAZAR CRÉDITO</span>'
        explanation = """
        **Recomendación: RECHAZAR**
        
        El cliente presenta indicadores de alto riesgo. Las variables financieras sugieren 
        una alta probabilidad de incumplimiento. Se recomienda no aprobar el crédito.
        """
        color = "error"

    return badge, explanation, color


# ============================================
# INTERFAZ PRINCIPAL
# ============================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Sistema de Evaluación de Crédito</h1>
        <p>Machine Learning en Producción - PECD UNI</p>
    </div>
    """, unsafe_allow_html=True)

    # Cargar modelo
    model, feature_names, metadata = load_model()

    if model is None:
        st.stop()

    # Sidebar - Información del modelo
    with st.sidebar:
        st.markdown("### 📋 Información del Modelo")

        with st.expander("📊 Métricas", expanded=True):
            st.metric("AUC-ROC", f"{metadata['metrics']['auc_test']:.4f}")
            st.metric("Features Totales", metadata['n_features'])
            st.metric("Train Samples",
                      f"{metadata['metrics']['n_train_samples']:,}")

        with st.expander("⚙️ Configuración"):
            st.write(f"**Modelo:** {metadata['model_type']}")
            st.write(
                f"**LightGBM:** v{metadata['training_info']['lightgbm_version']}")
            st.write(f"**Fecha:** {metadata['training_info']['date'][:10]}")

        st.markdown("---")
        st.markdown("### 🎚️ Umbrales de Decisión")
        st.write(f"🟢 Aprobar: < {THRESHOLD_LOW*100:.0f}%")
        st.write(
            f"🟡 Revisar: {THRESHOLD_LOW*100:.0f}% - {THRESHOLD_HIGH*100:.0f}%")
        st.write(f"🔴 Rechazar: > {THRESHOLD_HIGH*100:.0f}%")

        st.markdown("---")
        st.markdown("### 🎓 PECD - UNI")
        st.caption("Programa de Especialización en Ciencia de Datos")

    # Contenido principal - Dos columnas
    col_input, col_result = st.columns([1, 1.2])

    # Columna de inputs
    with col_input:
        st.markdown("### 📝 Datos del Solicitante")
        st.markdown(
            "Complete los siguientes campos para evaluar el riesgo crediticio:")

        input_values = {}

        for feat_name, config in FEATURE_CONFIG.items():
            st.markdown(f"**{config['label']}**")
            st.caption(config['description'])

            if isinstance(config['default'], float):
                input_values[feat_name] = st.slider(
                    label=feat_name,
                    min_value=float(config['min']),
                    max_value=float(config['max']),
                    value=float(config['default']),
                    step=float(config['step']),
                    help=config['help'],
                    label_visibility="collapsed"
                )
            else:
                input_values[feat_name] = st.slider(
                    label=feat_name,
                    min_value=int(config['min']),
                    max_value=int(config['max']),
                    value=int(config['default']),
                    step=int(config['step']),
                    help=config['help'],
                    label_visibility="collapsed"
                )
            st.markdown("")

        predict_button = st.button(
            "🔮 **EVALUAR RIESGO**", use_container_width=True, type="primary")

    # Columna de resultados
    with col_result:
        if predict_button:
            # Preparar datos - rellenar las features no usadas con 0
            input_df = pd.DataFrame({feat: [0.0] for feat in feature_names})

            for feat, val in input_values.items():
                if feat in input_df.columns:
                    input_df[feat] = val

            with st.spinner("Analizando perfil crediticio..."):
                probability = model.predict_proba(input_df)[0, 1]

            st.markdown("### 📊 Resultado del Análisis")

            # Gauge
            gauge_fig = create_gauge_chart(probability)
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Decisión
            badge_html, explanation, color = get_decision_badge(probability)

            st.markdown(f"""
            <div style="text-align: center; margin: 20px 0;">
                {badge_html}
            </div>
            """, unsafe_allow_html=True)

            if color == "success":
                st.success(explanation)
            elif color == "warning":
                st.warning(explanation)
            else:
                st.error(explanation)

            st.markdown("---")
            impact_fig = create_feature_impact_chart(
                input_values, FEATURE_CONFIG)
            st.plotly_chart(impact_fig, use_container_width=True)

            with st.expander("🔍 Ver datos técnicos"):
                st.json({
                    "probabilidad_default": round(probability, 4),
                    "decision": "APROBAR" if probability < THRESHOLD_LOW else ("REVISAR" if probability < THRESHOLD_HIGH else "RECHAZAR"),
                    "inputs": input_values,
                    "threshold_low": THRESHOLD_LOW,
                    "threshold_high": THRESHOLD_HIGH
                })
        else:
            st.markdown("### 📊 Resultado del Análisis")
            st.info(
                "👈 Complete los datos del solicitante y presione **EVALUAR RIESGO** para obtener el análisis.")

            st.markdown("""
            <div style="text-align: center; padding: 50px; color: #6c757d;">
                <h1>🏦</h1>
                <p>Esperando datos del solicitante...</p>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>🎓 <strong>Machine Learning Supervisado</strong> - Programa de Especialización en Ciencia de Datos</p>
        <p>Universidad Nacional de Ingeniería (UNI) - 2025</p>
        <p><em>Este es un modelo de demostración con fines educativos.</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

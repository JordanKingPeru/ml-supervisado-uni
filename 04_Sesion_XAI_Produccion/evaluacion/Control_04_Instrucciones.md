# 📝 Control N° 4: XAI, Serialización y Deployment

**Curso:** Machine Learning Supervisado - PECD UNI  
**Docente:** Jordan King Rodriguez Mallqui  
**Sesión:** 04 - De la Caja Negra a la Realidad Productiva  
**Fecha de entrega:** _14/12/2025_  
**Modalidad:** Individual o en parejas

---

## 🎯 Objetivo

Aplicar técnicas de **Explainable AI (SHAP)**, **validación cruzada avanzada**, **serialización de modelos** y **deployment** para llevar un modelo de Machine Learning desde el notebook hasta una aplicación funcional.

---

## 📋 Instrucciones Generales

### 1. Selección del Dataset

Puedes usar el **mismo dataset de los controles anteriores** o elegir uno nuevo. Se recomienda mantener consistencia para poder comparar el progreso a lo largo del curso.

#### 🔹 Opción A: Datasets Sugeridos

| Dataset | Descripción | Por qué es bueno para XAI | Enlace |
|---------|-------------|---------------------------|--------|
| **Credit Risk** | Predicción de default | Regulaciones exigen explicabilidad | [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) |
| **Employee Attrition** | Predicción de rotación | RRHH necesita entender causas | [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| **Heart Disease** | Diagnóstico médico | Médicos requieren explicación | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease) |
| **Telco Churn** | Fuga de clientes | Marketing necesita actionable insights | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Loan Default** | Préstamos | Compliance bancario | [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |

#### 🔹 Opción B: Dataset Propio

Requisitos:
- Problema de **clasificación binaria** o **regresión**
- Disponible en **repositorio público**
- Al menos **500 registros** y **5+ features**
- Contexto donde la **explicabilidad sea relevante**

---

## 📦 Entregables

### A. Notebook Jupyter Principal

#### **1. Introducción y Contexto** (5 pts)
- Descripción del problema
- ¿Por qué es importante explicar las predicciones en este contexto?
- Stakeholders que consumirán las explicaciones

#### **2. Validación Cruzada Avanzada** (15 pts)

Implementar al menos **2 estrategias de cross-validation**:

| Estrategia | Cuándo Usarla |
|------------|---------------|
| `StratifiedKFold` | Clasificación con clases desbalanceadas |
| `TimeSeriesSplit` | Datos con componente temporal |
| `GroupKFold` | Múltiples observaciones por entidad |
| `RepeatedStratifiedKFold` | Mayor robustez estadística |

Para cada estrategia:
- Justificar por qué es apropiada para tu dataset
- Reportar métricas con media ± desviación estándar
- Comparar resultados entre estrategias

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
print(f"AUC-ROC: {scores.mean():.4f} ± {scores.std():.4f}")
```

#### **3. Explainable AI con SHAP** (30 pts)

Implementar un análisis completo de SHAP que incluya:

**3.1 Interpretación Global (15 pts)**
- **Bar Plot:** Importancia promedio de features
- **Beeswarm Plot:** Distribución de impactos por feature
- **Heatmap:** Visualización matricial de SHAP values

```python
import shap

# Crear explainer (usar TreeExplainer para modelos de árboles)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Visualizaciones globales
shap.plots.bar(shap_values)
shap.plots.beeswarm(shap_values)
shap.plots.heatmap(shap_values)
```

**3.2 Interpretación Local (10 pts)**
- **Waterfall Plot:** Explicación de 2-3 predicciones individuales
- **Decision Plot:** Trayectoria de decisión
- Narrativa: "¿Por qué este cliente tiene alta/baja probabilidad?"

```python
# Explicación de una predicción específica
shap.plots.waterfall(shap_values[0])
shap.plots.decision(explainer.expected_value, shap_values.values[:10])
```

**3.3 Análisis de Dependencias (5 pts)**
- **Scatter Plot:** Relación SHAP vs valor de feature para top 2 variables
- Identificar relaciones no lineales e interacciones

```python
shap.plots.scatter(shap_values[:, "feature_name"])
```

#### **4. Serialización del Modelo** (15 pts)

Implementar al menos **2 métodos de serialización**:

| Método | Formato | Ventaja |
|--------|---------|---------|
| Pickle | `.pkl` | Simple, nativo Python |
| Joblib | `.joblib` | Comprimido, eficiente para arrays |
| LightGBM/XGBoost Native | `.txt` / `.json` | Portable, inspeccionable |
| ONNX | `.onnx` | Cross-platform (bonus) |

Para cada método:
- Guardar modelo entrenado
- Cargar y verificar predicciones
- Comparar tamaño de archivo

```python
import joblib

# Guardar
joblib.dump(model, 'model.joblib', compress=3)

# Cargar
model_loaded = joblib.load('model.joblib')

# Verificar
assert (model.predict(X_test) == model_loaded.predict(X_test)).all()
```

**Guardar metadatos** en JSON:
```python
import json

metadata = {
    "model_name": "credit_scoring_v1",
    "version": "1.0.0",
    "created_at": "2025-12-07",
    "features": list(X.columns),
    "metrics": {"auc": 0.95, "f1": 0.88},
    "threshold": 0.35
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

#### **5. Conclusiones del Análisis** (5 pts)
- Top 3 variables más importantes y su interpretación
- Insights de negocio derivados de SHAP
- Recomendaciones para stakeholders no técnicos

---

### B. Aplicación de Deployment (30 pts)

Crear una **aplicación Streamlit** que permita:

#### **5.1 Funcionalidad Básica (15 pts)**
- Cargar el modelo serializado
- Inputs para las variables del modelo (sliders, selectbox, etc.)
- Mostrar la predicción (probabilidad y clase)

#### **5.2 Explicabilidad Integrada (10 pts)**
- Mostrar explicación SHAP de la predicción individual
- Visualización del waterfall plot o similar
- Texto explicativo para usuarios no técnicos

#### **5.3 Interfaz de Usuario (5 pts)**
- Título y descripción del problema
- Organización clara de secciones
- Indicador visual de riesgo (semáforo, gauge, etc.)

### Estructura de archivos esperada:

```
entrega/
├── notebooks/
│   └── Control04_Apellido_Nombre.ipynb
├── app/
│   ├── app.py                    # Aplicación Streamlit
│   ├── models/
│   │   ├── model.joblib          # Modelo serializado
│   │   └── model_metadata.json   # Metadatos
│   └── requirements.txt          # Dependencias
└── README.md                     # Instrucciones de ejecución
```

### Código base para la app:

```python
# app.py
import streamlit as st
import joblib
import shap
import pandas as pd

# Cargar modelo
@st.cache_resource
def load_model():
    return joblib.load('models/model.joblib')

model = load_model()

st.title("🏦 Predictor de Riesgo Crediticio")
st.markdown("Ingrese los datos del cliente para obtener la predicción")

# Inputs
col1, col2 = st.columns(2)
with col1:
    edad = st.slider("Edad", 18, 80, 35)
    ingresos = st.number_input("Ingresos Mensuales", 1000, 50000, 5000)
with col2:
    deuda = st.slider("Ratio Deuda/Ingreso", 0.0, 1.0, 0.3)
    historial = st.selectbox("Historial Crediticio", ["Bueno", "Regular", "Malo"])

# Predicción
if st.button("🔮 Calcular Riesgo"):
    # Preparar datos
    X_new = pd.DataFrame([[edad, ingresos, deuda, historial]], 
                         columns=['edad', 'ingresos', 'deuda', 'historial'])
    
    # Predecir
    proba = model.predict_proba(X_new)[0][1]
    
    # Mostrar resultado
    st.metric("Probabilidad de Default", f"{proba:.1%}")
    
    if proba > 0.5:
        st.error("⚠️ ALTO RIESGO")
    else:
        st.success("✅ BAJO RIESGO")
```

---

## ⚠️ Criterios de Evaluación

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| **Reproducibilidad** | 10 | Notebook y app ejecutables |
| **Validación CV** | 15 | Al menos 2 estrategias correctamente implementadas |
| **SHAP Global** | 15 | Bar, Beeswarm, Heatmap con interpretación |
| **SHAP Local** | 15 | Waterfall, Decision Plot con narrativa |
| **Serialización** | 15 | 2+ métodos con verificación y metadatos |
| **App Streamlit** | 20 | Funcional con explicabilidad integrada |
| **Conclusiones** | 10 | Insights de negocio claros |
| **TOTAL** | **100** | |

---

## 🚫 Errores que Penalizan

- ❌ SHAP sin interpretación (solo código, sin análisis) (-15 pts)
- ❌ App que no ejecuta (-20 pts)
- ❌ No guardar metadatos del modelo (-5 pts)
- ❌ Usar solo un método de serialización (-5 pts)
- ❌ Validación con solo train_test_split (-10 pts)
- ❌ No incluir instrucciones para ejecutar la app (-5 pts)

---

## 📤 Formato de Entrega

1. **Archivos:** 
   - `Control04_Apellido_Nombre.ipynb`
   - Carpeta `app/` con la aplicación
   - `README.md` con instrucciones

2. **Plataforma:** [Canvas](https://canvas.instructure.com/courses/12906015)

3. **Formato:** ZIP con toda la estructura de carpetas

### Estructura del README.md:

```markdown
# Control 04 - [Tu Nombre]

## Descripción
[Breve descripción del proyecto]

## Requisitos
```bash
pip install -r requirements.txt
```

## Ejecutar la aplicación
```bash
cd app
streamlit run app.py
```

## Estructura del proyecto
[Explicar carpetas y archivos]
```

---

## 💡 Tips para un Buen Trabajo

1. **SHAP es para explicar, no solo mostrar:** Cada gráfico debe tener un párrafo de interpretación
2. **La app debe ser usable:** Piensa en un usuario no técnico
3. **Metadatos son críticos:** En producción real, sin metadatos el modelo es inútil
4. **Prueba tu app:** Asegúrate de que funcione antes de entregar
5. **README completo:** El evaluador debe poder ejecutar todo sin preguntarte

### Código de Referencia: SHAP con LightGBM

```python
import lightgbm as lgb
import shap

# Entrenar modelo
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# SHAP TreeExplainer (más rápido para modelos de árboles)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Para clasificación binaria, usar la clase positiva
# shap_values tiene shape (n_samples, n_features) si es binario
# o lista de arrays si es multiclase
```

---

## 📚 Recursos de Apoyo

### 📂 Repositorio del Curso
Todo el material está disponible en:

🔗 **[https://github.com/JordanKingPeru/ml-supervisado-uni](https://github.com/JordanKingPeru/ml-supervisado-uni)**

### 📓 Notebooks de la Sesión 4
- `01_Advanced_Validation.ipynb` - Estrategias de cross-validation
- `02_Explainable_AI_SHAP.ipynb` - Tutorial completo de SHAP
- `03_Model_Serialization.ipynb` - Métodos de serialización

### 📂 Aplicaciones de Ejemplo
- `app/app.py` - Aplicación completa de referencia
- `app/app_basic.py` - Versión mínima (~80 líneas)

### 🔗 Documentación Oficial
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Joblib Documentation](https://joblib.readthedocs.io/)
- [Cross-Validation - Scikit-Learn](https://scikit-learn.org/stable/modules/cross_validation.html)

### 📖 Lecturas Recomendadas
- *"Interpretable Machine Learning"* - Christoph Molnar (Gratis online)
- *"A Unified Approach to Interpreting Model Predictions"* - Lundberg & Lee (2017)

---

## ❓ Preguntas Frecuentes

**P: ¿Puedo usar otra librería de XAI como LIME?**  
R: Sí, pero SHAP debe ser el principal. LIME puede ser complementario (+3 pts bonus).

**P: ¿Qué hago si SHAP tarda mucho?**  
R: Usa `shap.TreeExplainer` para modelos de árboles. Reduce el tamaño del test set si es necesario.

**P: ¿Puedo usar Gradio en vez de Streamlit?**  
R: Sí, se acepta como alternativa válida.

**P: ¿Es obligatorio el SHAP en la app?**  
R: Sí, al menos un waterfall plot o force plot de la predicción individual.

**P: ¿Cómo subo la app a la nube?**  
R: No es obligatorio, pero si despliegas en Streamlit Cloud, Hugging Face Spaces o Render, ganas bonus (+5 pts).

---

## 🏆 Bonus Points (+15 pts máximo)

- **+5 pts:** Desplegar la app en la nube (Streamlit Cloud, HuggingFace, etc.)
- **+5 pts:** Implementar ONNX para serialización cross-platform
- **+3 pts:** Incluir LIME además de SHAP
- **+2 pts:** Implementar logging de predicciones en la app

---

## 🎓 Nota Final del Curso

Este es el **último control** del curso de Machine Learning Supervisado. El objetivo es que demuestres dominio del ciclo completo:

```
Datos → Pipeline → Modelo → Optimización → Explicabilidad → Producción
```

¡Muestra todo lo aprendido! 🚀

---

¡Éxitos! 🎉🔍📦


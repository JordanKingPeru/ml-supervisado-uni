# 📝 Control N° 1: Pipelines y Baselines

**Curso:** Machine Learning Supervisado - PECD UNI  
**Docente:** Jordan King Rodriguez Mallqui  
**Sesión:** 01 - Pipelines y Baselines  
**Fecha de entrega:** _(definir en cada edición)_  
**Modalidad:** Individual o en parejas

---

## 🎯 Objetivo

Aplicar los conceptos de **Pipelines de Scikit-Learn**, **construcción de baselines** y **buenas prácticas** para evitar **Data Leakage** en un problema de clasificación o regresión de tu elección.

---

## 📋 Instrucciones Generales

### 1. Selección del Dataset

Elige **UNO** de los siguientes datasets públicos o propón uno propio que cumpla los requisitos:

#### 🔹 Opción A: Datasets Sugeridos (Clasificación)

| Dataset | Descripción | Enlace |
|---------|-------------|--------|
| **Titanic** | Predicción de supervivencia | [Kaggle](https://www.kaggle.com/c/titanic/data) |
| **Heart Disease** | Predicción de enfermedad cardíaca | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease) |
| **Bank Marketing** | Predicción de suscripción a depósito | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) |
| **Adult Income** | Predicción de ingreso >50K | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/adult) |
| **Diabetes** | Predicción de diabetes en pacientes | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |

#### 🔹 Opción B: Datasets Sugeridos (Regresión)

| Dataset | Descripción | Enlace |
|---------|-------------|--------|
| **House Prices** | Predicción de precio de viviendas | [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| **California Housing** | Precio medio de viviendas en California | [Scikit-Learn](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) |
| **Auto MPG** | Predicción de consumo de combustible | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/auto+mpg) |
| **Wine Quality** | Predicción de calidad del vino | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) |

#### 🔹 Opción C: Dataset Propio

Puedes elegir cualquier dataset que:
- Esté disponible en un **repositorio público** (Kaggle, UCI, OpenML, GitHub, etc.)
- Tenga al menos **500 registros** y **5 features**
- Sea un problema de **clasificación binaria/multiclase** o **regresión**
- Incluyas el enlace de descarga en tu notebook, este debe de poder ejecutarse de inicio a fin, es decir debe de ser replicable.

---

## 📦 Entregables

### Notebook Jupyter con las siguientes secciones:

#### **1. Introducción y Contexto del Negocio** (10 pts)
- Descripción del problema y su relevancia
- Definición de la variable objetivo (target)
- Pregunta de negocio que intentas responder

#### **2. Carga y Exploración de Datos (EDA)** (15 pts)
- Carga del dataset desde fuente pública
- Análisis descriptivo básico (shape, tipos, estadísticas)
- Identificación de valores faltantes
- Visualización de distribuciones y relaciones clave

#### **3. Preprocesamiento con Pipeline** (25 pts)
- Construcción de un `ColumnTransformer` que maneje:
  - Variables numéricas (imputación + escalado)
  - Variables categóricas (imputación + encoding)
- Pipeline completo que incluya preprocesamiento + modelo
- **IMPORTANTE:** El preprocesamiento debe estar dentro del pipeline para evitar Data Leakage

#### **4. Modelo Baseline** (20 pts)
- Implementación de al menos **2 modelos baseline**:
  - Modelo "dummy" (DummyClassifier o DummyRegressor)
  - Modelo simple (LogisticRegression / LinearRegression)
- Justificación de por qué estos modelos son buenos baselines
- Evaluación en conjunto de validación

#### **5. Evaluación y Métricas** (20 pts)
- División correcta train/test (sin data leakage)
- Métricas apropiadas según el tipo de problema:
  - **Clasificación:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Matriz de Confusión
  - **Regresión:** MSE, RMSE, MAE, R²
- Comparación de modelos en tabla resumen

#### **6. Conclusiones** (10 pts)
- Interpretación de resultados
- ¿El baseline es suficiente? ¿Por qué sí/no?
- Próximos pasos sugeridos para mejorar el modelo

---

## ⚠️ Criterios de Evaluación

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| **Reproducibilidad** | 10 | El notebook debe ejecutarse de principio a fin sin errores |
| **Código limpio** | 10 | Variables con nombres descriptivos, código comentado |
| **Sin Data Leakage** | 15 | Todo preprocesamiento dentro del pipeline |
| **Pipeline completo** | 25 | ColumnTransformer + Pipeline bien estructurado |
| **Evaluación correcta** | 20 | Métricas apropiadas |
| **Análisis y conclusiones** | 20 | Interpretación de negocio coherente |
| **TOTAL** | **100** | |

---

## 🚫 Errores que Penalizan

- ❌ Hacer `fit_transform` en todo el dataset antes de split (-15 pts)
- ❌ Escalar o imputar fuera del pipeline (-10 pts)
- ❌ No usar `random_state` en splits y modelos (-5 pts)
- ❌ Notebook que no ejecuta de principio a fin (-20 pts)
- ❌ Dataset no accesible públicamente (-10 pts)

---

## 📤 Formato de Entrega

1. **Archivo:** `Control01_Apellido_Nombre.ipynb`
2. **Plataforma:** [_Canvas_](https://canvas.instructure.com/courses/12906015)


### Estructura sugerida del notebook:

```
1. Título y datos del alumno
2. Introducción
3. Carga de datos
4. EDA
5. Preprocesamiento (Pipeline)
6. Modelado y Baseline
7. Evaluación
8. Conclusiones
9. Referencias
```

---

## 💡 Tips para un Buen Trabajo

1. **Comienza simple:** Un pipeline básico que funcione es mejor que uno complejo con errores
2. **Documenta tu razonamiento:** Explica por qué tomas cada decisión
3. **Verifica el leakage:** Pregúntate "¿estoy usando información del test en el train?"
4. **Compara con el dummy:** Si tu modelo no supera al dummy, algo está mal
5. **Guarda tu pipeline:** Usa `joblib.dump()` para guardar el modelo entrenado

---

## 📚 Recursos de Apoyo

### 📂 Repositorio del Curso
Puedes utilizar como referencia todo el material disponible en el repositorio oficial del curso:

🔗 **[https://github.com/JordanKingPeru/ml-supervisado-uni](https://github.com/JordanKingPeru/ml-supervisado-uni)**

### 📓 Notebooks del Curso
- **[00_Dummy_Baselines_Intro.ipynb](../notebooks/00_Dummy_Baselines_Intro.ipynb)** - Tutorial completo de DummyClassifier y DummyRegressor ⭐
- `01_Anti_Pattern.ipynb` - Errores comunes y cómo evitarlos
- `02_Pipelines_y_Baselines.ipynb` - Construcción de Pipelines profesionales

### 🔗 Documentación Oficial
- [DummyClassifier - Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
- [DummyRegressor - Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)
- [Documentación de Pipelines - Scikit-Learn](https://scikit-learn.org/stable/modules/compose.html)
- [ColumnTransformer - Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Cross-Validation - Scikit-Learn](https://scikit-learn.org/stable/modules/cross_validation.html)

---

## ❓ Preguntas Frecuentes

**P: ¿Puedo usar un dataset que no está en la lista?**  
R: Sí, siempre que sea público y cumpla los requisitos mínimos.

**P: ¿Puedo usar modelos más avanzados que Regresión Logística?**  
R: Puedes incluirlos como comparación adicional, pero el baseline debe ser simple.

**P: ¿Qué hago si mi dataset tiene muchos valores faltantes?**  
R: Documenta tu estrategia de imputación y justifícala. Es parte del aprendizaje.

**P: ¿Puedo trabajar en grupo?**  
R: Máximo 2 personas. Ambos deben entender todo el código.

---

¡Éxitos! 🚀

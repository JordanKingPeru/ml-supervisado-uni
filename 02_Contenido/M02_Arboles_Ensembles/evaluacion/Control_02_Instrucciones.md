# 📝 Control N° 2: Árboles de Decisión y Métodos Ensemble

**Curso:** Machine Learning Supervisado - PECD UNI  
**Docente:** Jordan King Rodriguez Mallqui  
**Sesión:** 02 - Árboles de Decisión y Ensembles  
**Fecha de entrega:** _(definir en cada edición)_  
**Modalidad:** Individual o en parejas

---

## 🎯 Objetivo

Aplicar y comparar **algoritmos no lineales** (Árboles de Decisión, Random Forest, Gradient Boosting, SVM, KNN) en un problema de clasificación o regresión, demostrando comprensión de sus fortalezas, debilidades y el impacto de sus hiperparámetros.

---

## 📋 Instrucciones Generales

### 1. Selección del Dataset

Puedes usar el **mismo dataset del Control N° 1** o elegir uno nuevo de las siguientes opciones:

#### 🔹 Opción A: Datasets Sugeridos (Clasificación)

| Dataset | Descripción | Enlace |
|---------|-------------|--------|
| **Telco Churn** | Predicción de fuga de clientes | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Credit Card Fraud** | Detección de fraude | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Mushroom** | Clasificación comestible/venenoso | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/mushroom) |
| **Breast Cancer** | Diagnóstico de cáncer | [Scikit-Learn](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) |
| **Customer Segmentation** | Segmentación de clientes | [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) |

#### 🔹 Opción B: Datasets Sugeridos (Regresión)

| Dataset | Descripción | Enlace |
|---------|-------------|--------|
| **Boston Housing** | Precio de viviendas | [Kaggle](https://www.kaggle.com/datasets/vikrishnan/boston-house-prices) |
| **Bike Sharing** | Demanda de bicicletas | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) |
| **Energy Efficiency** | Consumo energético de edificios | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/energy+efficiency) |
| **Insurance Charges** | Predicción de costos médicos | [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) |

#### 🔹 Opción C: Dataset Propio

Requisitos:
- Disponible en **repositorio público** (Kaggle, UCI, OpenML, GitHub, etc.)
- Al menos **500 registros** y **5 features**
- Problema de **clasificación** o **regresión**
- Enlace de descarga incluido (notebook replicable)

---

## 📦 Entregables

### Notebook Jupyter con las siguientes secciones:

#### **1. Introducción y Contexto** (5 pts)
- Descripción breve del problema
- Justificación de por qué un modelo no lineal podría ser útil

#### **2. Carga y Preprocesamiento** (10 pts)
- Carga del dataset
- Pipeline de preprocesamiento (reutilizar lo aprendido en Sesión 1)
- División train/test correcta

#### **3. Entrenamiento de Modelos** (30 pts)

Entrenar al menos **4 de los siguientes modelos**:

| Modelo | Clase en Scikit-Learn |
|--------|----------------------|
| Árbol de Decisión | `DecisionTreeClassifier` / `DecisionTreeRegressor` |
| Random Forest | `RandomForestClassifier` / `RandomForestRegressor` |
| Gradient Boosting | `GradientBoostingClassifier` / `GradientBoostingRegressor` |
| XGBoost | `XGBClassifier` / `XGBRegressor` |
| SVM | `SVC` / `SVR` |
| KNN | `KNeighborsClassifier` / `KNeighborsRegressor` |

Para cada modelo:
- Entrenar con hiperparámetros por defecto
- Documentar brevemente qué hace el algoritmo

#### **4. Comparación de Modelos** (20 pts)
- Tabla comparativa con métricas de todos los modelos
- Gráfico de barras comparando rendimiento
- Análisis: ¿Cuál modelo tiene mejor desempeño? ¿Por qué?

#### **5. Ajuste Manual de Hiperparámetros** (20 pts)

Seleccionar el **mejor modelo** de la comparación y realizar:
- Probar **manualmente** al menos **3 configuraciones diferentes** de hiperparámetros
- **Justificar** por qué elegiste esos valores (basándote en la teoría vista en clase)
- Comparar resultados de cada configuración
- Reportar mejora obtenida (si la hay)

> **📝 Nota:** En la siguiente sesión veremos técnicas automáticas como `GridSearchCV` y `RandomizedSearchCV`. Por ahora, el objetivo es que entiendas el **efecto de cada hiperparámetro** mediante experimentación manual.

Ejemplo de hiperparámetros a explorar:

| Modelo | Hiperparámetros Sugeridos |
|--------|---------------------------|
| Decision Tree | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| Random Forest | `n_estimators`, `max_depth`, `max_features` |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth` |
| SVM | `C`, `kernel`, `gamma` |
| KNN | `n_neighbors`, `weights`, `metric` |

#### **6. Interpretabilidad** (10 pts)
- Para modelos basados en árboles: **Feature Importance**
- Visualización del árbol (si aplica, con `plot_tree` o similar)
- Interpretación: ¿Qué variables son más importantes para el modelo?

#### **7. Conclusiones** (5 pts)
- ¿Qué modelo recomiendas para producción y por qué?
- Trade-offs observados (accuracy vs interpretabilidad, tiempo de entrenamiento, etc.)
- Próximos pasos sugeridos

---

## ⚠️ Criterios de Evaluación

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| **Reproducibilidad** | 10 | Notebook ejecutable de principio a fin |
| **Pipeline correcto** | 10 | Preprocesamiento dentro del pipeline, sin data leakage |
| **Modelos entrenados** | 30 | Al menos 4 modelos diferentes correctamente implementados |
| **Comparación justa** | 20 | Métricas apropiadas, mismo split para todos |
| **Tuning de hiperparámetros** | 20 | Experimentación manual con justificación |
| **Interpretabilidad** | 10 | Feature importance y/o visualización del árbol |
| **TOTAL** | **100** | |

---

## 🚫 Errores que Penalizan

- ❌ Comparar modelos con diferentes splits de datos (-10 pts)
- ❌ No usar `random_state` en modelos y splits (-5 pts)
- ❌ No justificar los cambios de hiperparámetros (-10 pts)
- ❌ Notebook que no ejecuta de principio a fin (-20 pts)
- ❌ Solo entrenar 2 modelos o menos (-15 pts)
- ❌ No incluir Feature Importance cuando aplica (-5 pts)

---

## 📤 Formato de Entrega

1. **Archivo:** `Control02_Apellido_Nombre.ipynb`
2. **Plataforma:** [Canvas](https://canvas.instructure.com/courses/12906015)

### Estructura sugerida del notebook:

```
1. Título y datos del alumno
2. Introducción
3. Carga y Preprocesamiento
4. Entrenamiento de Modelos (4+ modelos)
5. Comparación y Análisis
6. Ajuste Manual de Hiperparámetros
7. Interpretabilidad (Feature Importance)
8. Conclusiones
9. Referencias
```

---

## 💡 Tips para un Buen Trabajo

1. **Reutiliza tu Pipeline:** El preprocesamiento del Control 1 puede servir aquí
2. **Usa `cross_val_score`:** Para una comparación más robusta
3. **Cuidado con el tiempo:** Algunos modelos (SVM, GridSearch extenso) pueden tardar mucho
4. **Documenta tus decisiones:** ¿Por qué elegiste esos rangos de hiperparámetros?
5. **Visualiza:** Un gráfico vale más que mil números

### Código de Referencia: Comparación Rápida

```python
from sklearn.model_selection import cross_val_score

modelos = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

resultados = []
for nombre, modelo in modelos.items():
    scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')
    resultados.append({
        'Modelo': nombre,
        'CV Mean': scores.mean(),
        'CV Std': scores.std()
    })

pd.DataFrame(resultados).sort_values('CV Mean', ascending=False)
```

---

## 📚 Recursos de Apoyo

### 📂 Repositorio del Curso
Todo el material está disponible en:

🔗 **[https://github.com/JordanKingPeru/ml-supervisado-uni](https://github.com/JordanKingPeru/ml-supervisado-uni)**

### 📓 Notebooks de la Sesión 2
- `00a_Arboles_Decision_Intro.ipynb` - Fundamentos de Árboles de Decisión
- `00b_Random_Forest_Intro.ipynb` - Random Forest y Bagging
- `00c_Gradient_Boosting_Intro.ipynb` - Gradient Boosting
- `00d_SVM_Intro.ipynb` - Support Vector Machines
- `00e_KNN_Intro.ipynb` - K-Nearest Neighbors
- `01_Algoritmos_No_Lineales.ipynb` - Comparación de algoritmos
- `02_Arena_Combate.ipynb` - Competencia de modelos

### 🔗 Documentación Oficial
- [Decision Trees - Scikit-Learn](https://scikit-learn.org/stable/modules/tree.html)
- [Random Forest - Scikit-Learn](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Gradient Boosting - Scikit-Learn](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- [SVM - Scikit-Learn](https://scikit-learn.org/stable/modules/svm.html)
- [KNN - Scikit-Learn](https://scikit-learn.org/stable/modules/neighbors.html)
- [GridSearchCV - Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) *(referencia para la siguiente sesión)*

### 📋 Cheatsheet de Hiperparámetros
Consulta el archivo `recursos/cheatsheets/hyperparameters_cheatsheet.md` del repositorio.

---

## ❓ Preguntas Frecuentes

**P: ¿Puedo usar el mismo dataset del Control 1?**  
R: Sí, de hecho es recomendable para comparar modelos lineales vs no lineales.

**P: ¿Es obligatorio usar XGBoost?**  
R: No, pero si lo usas ganas puntos extra por exploración (+5 pts bonus).

**P: ¿Qué hago si no sé qué valores probar para los hiperparámetros?**  
R: Revisa los notebooks de clase y el cheatsheet de hiperparámetros. Empieza con valores pequeños y ve aumentando.

**P: ¿Puedo usar LightGBM o CatBoost?**  
R: Sí, se consideran como alternativas válidas a Gradient Boosting.

**P: ¿Es necesario visualizar todos los árboles?**  
R: No, solo visualiza uno representativo (ej: un árbol del Random Forest o el Decision Tree simple).

---

## 🏆 Bonus Points (+10 pts máximo)

- **+5 pts:** Usar XGBoost, LightGBM o CatBoost
- **+3 pts:** Incluir curvas de aprendizaje (learning curves)
- **+2 pts:** Análisis de overfitting con diferentes `max_depth`

---

¡Éxitos! 🚀🌳

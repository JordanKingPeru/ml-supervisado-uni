# 📝 Control N° 3: Optimización y Calibración de Modelos

**Curso:** Machine Learning Supervisado - PECD UNI  
**Docente:** Jordan King Rodriguez Mallqui  
**Sesión:** 03 - Optimización y Calibración   
**Modalidad:** Individual

---

## 🎯 Objetivo

Aplicar técnicas de **optimización de hiperparámetros** (Optuna, GridSearchCV, RandomizedSearchCV), **calibración de probabilidades**, **selección de umbrales** y **análisis de curvas de rentabilidad** para maximizar el valor de negocio de un modelo de clasificación.

---

## 📋 Instrucciones Generales

### 1. Selección del Dataset

Puedes usar el **mismo dataset de los controles anteriores** o elegir uno nuevo. Para este control, se recomienda usar un dataset de **clasificación binaria** donde el contexto de negocio permita definir costos/beneficios.

#### 🔹 Opción A: Datasets Sugeridos (Clasificación con Contexto de Negocio)

| Dataset | Descripción | Contexto de Negocio | Enlace |
|---------|-------------|---------------------|--------|
| **Telco Churn** | Predicción de fuga de clientes | Costo de retención vs valor del cliente | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Credit Risk** | Predicción de default crediticio | Pérdida por default vs beneficio por interés | [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) |
| **Bank Marketing** | Suscripción a depósito | Costo de llamada vs valor del depósito | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) |
| **Employee Attrition** | Predicción de rotación laboral | Costo de reclutamiento vs retención | [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| **Insurance Cross-Sell** | Predicción de venta cruzada | Costo de campaña vs prima de seguro | [Kaggle](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction) |

#### 🔹 Opción B: Dataset Propio

Requisitos:
- Problema de **clasificación binaria**
- Disponible en **repositorio público**
- Al menos **1,000 registros** (para que Optuna tenga suficientes datos)
- Definir claramente un **contexto de negocio** con costos/beneficios

---

## 📦 Entregables

### Notebook Jupyter con las siguientes secciones:

#### **1. Introducción y Contexto de Negocio** (10 pts)
- Descripción del problema
- Definición de la **matriz de costos/beneficios**:
  - Beneficio de un True Positive (TP)
  - Costo de un False Positive (FP)
  - Costo de un False Negative (FN)
  - Beneficio/Costo de un True Negative (TN)
- Ejemplo: En un modelo de churn, ¿cuánto cuesta perder un cliente vs cuánto cuesta una campaña de retención fallida?

#### **2. Preprocesamiento y Modelo Base** (10 pts)
- Pipeline de preprocesamiento (reutilizar de controles anteriores)
- Entrenar un modelo base (ej: LightGBM con hiperparámetros por defecto)
- Métricas iniciales de referencia

#### **3. Optimización con Optuna** (30 pts)

Implementar un estudio de Optuna que:
- Defina un **espacio de búsqueda** de al menos **5 hiperparámetros**
- Use **cross-validation** dentro de la función objetivo
- Ejecute al menos **50 trials**
- Incluya **pruning** para eficiencia

```python
# Ejemplo de estructura esperada
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        # ... más hiperparámetros
    }
    
    model = LGBMClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

Documentar:
- Visualizaciones de Optuna (`plot_optimization_history`, `plot_param_importances`)
- Mejores hiperparámetros encontrados
- Comparación con modelo base

#### **4. Calibración de Probabilidades** (15 pts)

Aplicar al menos **2 métodos de calibración**:
- `CalibratedClassifierCV` con método `'isotonic'`
- `CalibratedClassifierCV` con método `'sigmoid'` (Platt Scaling)

Evaluar la calibración con:
- **Diagrama de Calibración** (`CalibrationDisplay`)
- **Brier Score** antes y después de calibrar
- Análisis: ¿Mejoró la calibración? ¿Cómo afectó otras métricas?

#### **5. Selección de Umbral Óptimo** (15 pts)

Implementar selección de umbral basada en:
- **Maximización de F1-Score** (umbral estándar)
- **Maximización del beneficio esperado** (usando la matriz de costos definida)

```python
# Ejemplo: Encontrar umbral que maximiza beneficio
def calcular_beneficio(y_true, y_pred, umbral, beneficios):
    y_pred_class = (y_pred >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    
    beneficio_total = (
        tp * beneficios['TP'] +
        tn * beneficios['TN'] -
        fp * beneficios['FP'] -
        fn * beneficios['FN']
    )
    return beneficio_total

# Buscar umbral óptimo
umbrales = np.arange(0.1, 0.9, 0.01)
beneficios_por_umbral = [calcular_beneficio(y_test, probs, u, matriz_costos) for u in umbrales]
umbral_optimo = umbrales[np.argmax(beneficios_por_umbral)]
```

Incluir:
- Gráfico de beneficio vs umbral
- Comparación de umbral 0.5 vs umbral óptimo
- Impacto en métricas (Precision, Recall, F1)

#### **6. Curva de Rentabilidad (Profit Curve)** (15 pts)

Implementar y analizar:
- **Curva Lift**
- **Curva de Ganancia Acumulada**
- **Curva de Beneficio**

```python
# Ejemplo: Curva de beneficio acumulado
def profit_curve(y_true, y_proba, costo_accion, beneficio_acierto):
    # Ordenar por probabilidad descendente
    sorted_indices = np.argsort(y_proba)[::-1]
    y_sorted = y_true.iloc[sorted_indices].values
    
    # Calcular beneficio acumulado
    n = len(y_true)
    beneficios = []
    for i in range(1, n+1):
        tp = y_sorted[:i].sum()
        fp = i - tp
        beneficio = tp * beneficio_acierto - i * costo_accion
        beneficios.append(beneficio)
    
    return np.arange(1, n+1) / n * 100, beneficios
```

Análisis requerido:
- ¿A qué porcentaje de la población deberíamos aplicar la acción?
- ¿Cuál es el beneficio máximo esperado?
- Comparación con estrategia aleatoria

#### **7. Conclusiones y Recomendaciones** (5 pts)
- Resumen de mejoras obtenidas en cada etapa
- Recomendación final: ¿Qué modelo, umbral y estrategia usar en producción?
- Limitaciones y próximos pasos

---

## ⚠️ Criterios de Evaluación

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| **Reproducibilidad** | 10 | Notebook ejecutable de principio a fin |
| **Contexto de negocio** | 10 | Matriz de costos bien definida y justificada |
| **Optuna implementado** | 30 | Estudio completo con visualizaciones |
| **Calibración correcta** | 15 | Dos métodos comparados con Brier Score |
| **Selección de umbral** | 15 | Umbral basado en negocio, no solo F1 |
| **Curvas de rentabilidad** | 15 | Profit curve con análisis de negocio |
| **Conclusiones** | 5 | Recomendación clara y fundamentada |
| **TOTAL** | **100** | |

---

## 🚫 Errores que Penalizan

- ❌ Optuna con menos de 30 trials (-10 pts)
- ❌ No usar cross-validation en Optuna (-15 pts)
- ❌ Calibrar en el mismo set de entrenamiento (data leakage) (-10 pts)
- ❌ No definir matriz de costos de negocio (-10 pts)
- ❌ Notebook que no ejecuta de principio a fin (-20 pts)
- ❌ Solo usar umbral 0.5 sin análisis (-10 pts)

---

## 📤 Formato de Entrega

1. **Archivo:** `Control03_Apellido_Nombre.ipynb`
2. **Plataforma:** [Canvas](https://canvas.instructure.com/courses/12906015)

### Estructura sugerida del notebook:

```
1. Título y datos del alumno
2. Introducción y Contexto de Negocio
3. Carga de Datos y Preprocesamiento
4. Modelo Base (Benchmark)
5. Optimización con Optuna
6. Calibración de Probabilidades
7. Selección de Umbral Óptimo
8. Curvas de Rentabilidad
9. Conclusiones y Recomendaciones
10. Referencias
```

---

## 💡 Tips para un Buen Trabajo

1. **Define el negocio primero:** Sin matriz de costos clara, la optimización no tiene sentido
2. **Optuna con paciencia:** 50+ trials para resultados estables
3. **Calibración separada:** Usar un conjunto de calibración diferente al de entrenamiento
4. **Visualiza todo:** Los stakeholders entienden mejor con gráficos
5. **El umbral importa:** Un modelo excelente con umbral incorrecto es inútil

### Código de Referencia: Visualización de Optuna

```python
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_contour
)

# Historial de optimización
fig1 = plot_optimization_history(study)
fig1.show()

# Importancia de hiperparámetros
fig2 = plot_param_importances(study)
fig2.show()
```

---

## 📚 Recursos de Apoyo

### 📂 Repositorio del Curso
Todo el material está disponible en:

🔗 **[https://github.com/JordanKingPeru/ml-supervisado-uni](https://github.com/JordanKingPeru/ml-supervisado-uni)**

### 📓 Notebooks de la Sesión 3
- `01_Hyperparameter_Optimization.ipynb` - Optuna y búsqueda de hiperparámetros
- `02_Probability_Calibration.ipynb` - Calibración de probabilidades
- `03_Threshold_Selection.ipynb` - Selección de umbrales
- `04_Profit_Curves.ipynb` - Curvas de rentabilidad

### 🔗 Documentación Oficial
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [CalibratedClassifierCV - Scikit-Learn](https://scikit-learn.org/stable/modules/calibration.html)
- [Probability Calibration - Scikit-Learn Guide](https://scikit-learn.org/stable/modules/calibration.html)
- [Classification Metrics - Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

### 📖 Lecturas Recomendadas
- *"Profit-Driven Business Analytics"* - Verbeke et al.
- *"Calibrating Probabilities"* - Platt (1999)

---

## ❓ Preguntas Frecuentes

**P: ¿Puedo usar GridSearchCV en vez de Optuna?**  
R: Sí, pero Optuna tiene puntos extra (+5 pts). Si usas GridSearchCV, asegúrate de que sea suficientemente exhaustivo.

**P: ¿Cómo defino los costos si no tengo información del negocio?**  
R: Puedes asumir valores razonables y documentar tus supuestos. Ejemplo: "Asumimos que perder un cliente cuesta 5x más que una campaña de retención fallida".

**P: ¿Cuántos trials son suficientes en Optuna?**  
R: Mínimo 50 para este control. En producción real, 100-500 trials son comunes.

**P: ¿Es obligatorio calibrar si el modelo ya da buenas probabilidades?**  
R: Sí, debes mostrar el proceso y comparar. A veces la calibración empeora modelos ya bien calibrados.

**P: ¿Puedo usar otros optimizadores como Hyperopt o Ray Tune?**  
R: Sí, se aceptan como alternativas válidas a Optuna.

---

## 🏆 Bonus Points (+10 pts máximo)

- **+5 pts:** Implementar Early Stopping con Optuna (pruning)
- **+3 pts:** Comparar `'isotonic'` vs `'sigmoid'` en detalle
- **+2 pts:** Incluir análisis de sensibilidad de la matriz de costos

---

¡Éxitos! 🚀⚙️


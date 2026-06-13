# 📘 Sesión 01: Ingeniería de ML y Baselines Robustos

**Fecha:** Sábado 29 de noviembre | **Horario:** 15:00 - 20:00
**Docente** Jordan King Rodriguez Mallqui

## 🎯 Objetivos de la Sesión
Al finalizar esta sesión, dejarás de limpiar datos celda por celda y serás capaz de:
1.  **Construir Pipelines de Scikit-Learn:** Empaquetar preprocesamiento y modelado en un solo objeto serializable.
2.  **Manejar Datos Reales:** Imputar nulos con estrategias multivariadas (KNN) y codificar categóricas sin explotar la memoria (Target Encoding).
3.  **Establecer un Baseline:** Crear un modelo lineal robusto (Regresión Logística Regularizada) que sirva como punto de comparación.
4.  **Detectar Data Leakage:** Identificar el error número 1 que hace que los proyectos fallen en producción.

## 📂 Estructura del Material
*   **`slides/`**: Presentación teórica de la sesión.
*   **`notebooks/`**:
    *   `01_Anti_Pattern.ipynb`: Ejemplo de cómo NO se debe programar (código spaghetti).
    *   `02_Pipelines_y_Baselines.ipynb`: El notebook maestro con las mejores prácticas.
    *   `03_Data_Leakage_Challenge.ipynb`: Ejercicio práctico de detección de fugas.
*   **`data/`**: Dataset `credit_scoring.csv`.

## 🛠️ Conceptos Clave
*   **Pipeline:** `sklearn.pipeline.Pipeline`
*   **ColumnTransformer:** `sklearn.compose.ColumnTransformer`
*   **Imputación:** `SimpleImputer`, `KNNImputer`
*   **Encoding:** `OneHotEncoder`, `TargetEncoder` (category_encoders)
*   **Modelo:** `LogisticRegression` (con `class_weight='balanced'`)
*   **Métrica:** `ROC-AUC`

## 📚 Tarea para la casa
Revisar el notebook maestro y tratar de replicar el pipeline con un dataset propio o uno de Kaggle (ej. Titanic o House Prices) para practicar la sintaxis.

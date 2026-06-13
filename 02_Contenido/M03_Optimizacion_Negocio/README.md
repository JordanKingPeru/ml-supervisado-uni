# 💰 Sesión 03: Tuning de Precisión y Rentabilidad

**Fecha:** Sábado 28 de Junio | **Horario:** 16:00 - 20:00

## 🎯 Objetivos de la Sesión
Aquí es donde separamos a los "Junior" (que solo buscan Accuracy) de los "Senior" (que buscan Rentabilidad).
1.  **Automatizar la Búsqueda:** Dejar de adivinar hiperparámetros manualmente. Usar **Optuna** para encontrar la configuración óptima.
2.  **Calibrar el Riesgo:** Entender que un modelo con AUC 0.90 puede mentir en sus probabilidades. Usar `CalibratedClassifierCV`.
3.  **Monetizar el Modelo:** Transformar una Matriz de Confusión en una **Tabla de Ganancias y Pérdidas (P&L)**.
4.  **Manejar Desbalance Moderno:** Usar pesos (`scale_pos_weight`) en lugar de re-muestreo destructivo (SMOTE).

## 📂 Estructura del Material
*   **`slides/`**: Presentación teórica.
*   **`notebooks/`**:
    *   `01_Optimization_and_Money.ipynb`: Notebook maestro que cubre Optuna, Calibración y Profit Curves.
*   **`data/`**: Dataset `credit_scoring.csv`.

## 🛠️ Conceptos Clave
*   **Optimización Bayesiana:** `optuna`
*   **Calibración:** `calibration_curve`, `CalibratedClassifierCV`, `Brier Score`
*   **Negocio:** Matriz de Costos, Profit Curve, Threshold Tuning
*   **Desbalance:** `scale_pos_weight` (LightGBM/XGBoost)

## 📚 Tarea para la casa
Tomar el modelo optimizado y calcular cuánto dinero ahorraría al banco si cada Falso Negativo cuesta $1000 y cada Falso Positivo cuesta $50.

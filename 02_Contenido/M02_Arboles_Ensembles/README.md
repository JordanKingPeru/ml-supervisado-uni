# 🌲 Sesión 02: La Revolución No-Lineal (Trees, SVM & Ensembles)

**Fecha:** Domingo 22 de Junio | **Horario:** 09:00 - 13:00

## 🎯 Objetivos de la Sesión
Al finalizar esta sesión, dejarás de ver los algoritmos como "cajas negras" y podrás:
1.  **Entender la No-Linealidad:** Por qué una Regresión Logística falla donde un Random Forest brilla.
2.  **Dominar los Árboles:** Controlar su crecimiento (Pruning) para evitar el overfitting.
3.  **Transición a Boosting:** Entender por qué XGBoost/LightGBM son el estándar en competencias (Kaggle).
4.  **Ubicar a los Clásicos (SVM/KNN):** Saber cuándo usarlos (ej. imputación, datasets pequeños) y cuándo no.

## 📂 Estructura del Material
*   **`slides/`**: Presentación teórica de la sesión.
*   **`notebooks/`**:
    *   `01_Algoritmos_No_Lineales.ipynb`: Notebook maestro con demos visuales (Moons, Trees, SVM) y teoría aplicada.
    *   `02_Arena_Combate.ipynb`: Taller competitivo (Random Forest vs LightGBM).
*   **`data/`**: Dataset `telco_churn.csv`.

## 🛠️ Conceptos Clave
*   **Decision Tree:** `DecisionTreeClassifier`, `plot_tree`
*   **Ensemble Learning:** Bagging (Random Forest) vs Boosting (Gradient Boosting)
*   **SOTA Algorithms:** `XGBoost`, `LightGBM`, `CatBoost`
*   **Algoritmos Geométricos:** `SVC` (SVM), `KNeighborsClassifier` (KNN)
*   **Visualización:** `mlxtend.plotting.plot_decision_regions`

## 📚 Tarea para la casa
Experimentar con los hiperparámetros de LightGBM (`num_leaves`, `learning_rate`) en el notebook de Arena de Combate para intentar superar el baseline.

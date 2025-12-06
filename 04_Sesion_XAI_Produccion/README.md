# 🚀 Sesión 04: De la Caja Negra a la Realidad Productiva

## 🎯 Objetivos
1.  **Validación Robusta:** Aprender a validar modelos en escenarios complejos (Series Temporales, Desbalance Extremo) usando `StratifiedKFold` y `TimeSeriesSplit`.
2.  **Explainable AI (XAI):** Abrir la "caja negra" de los modelos complejos usando **SHAP** (SHapley Additive exPlanations) para entender el *por qué* de las predicciones.
3.  **Producción (MLOps):** Serializar modelos con `joblib` y desplegarlos en una aplicación web interactiva usando **Streamlit**.

## 📂 Estructura
*   `notebooks/`:
    *   `01_Advanced_Validation.ipynb`: Técnicas avanzadas de validación cruzada.
    *   `02_Explainable_AI_SHAP.ipynb`: Interpretación de modelos con SHAP.
    *   `03_Model_Serialization.ipynb`: Entrenamiento final y guardado del modelo.
*   `app/`:
    *   `app.py`: Aplicación web Streamlit para demostración del modelo.
    *   `requirements.txt`: Dependencias de la app.
*   `data/`: Datasets utilizados (`credit_scoring.csv`, `DS_Compra.csv`).
*   `slides/`: Presentación de la sesión.

## 🛠️ Instalación
Para ejecutar la app de Streamlit:
```bash
pip install streamlit
cd app
streamlit run app.py
```

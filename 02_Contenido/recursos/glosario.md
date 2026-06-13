# 📚 Glosario de Términos - Machine Learning Supervisado

## A
*   **Accuracy (Exactitud):** Proporción de predicciones correctas sobre el total. Cuidado: engañosa en datasets desbalanceados.
*   **AUC-ROC:** Área bajo la curva ROC. Mide la capacidad del modelo para distinguir entre clases. 0.5 es azar, 1.0 es perfecto.

## B
*   **Bagging (Bootstrap Aggregating):** Técnica de ensamble que entrena múltiples modelos en subconjuntos aleatorios de datos (con reemplazo) y promedia sus resultados. Ejemplo: Random Forest.
*   **Bias (Sesgo):** Error debido a suposiciones erróneas en el algoritmo (ej. asumir linealidad cuando no la hay). Alto bias -> Underfitting.
*   **Boosting:** Técnica de ensamble secuencial donde cada modelo intenta corregir los errores del anterior. Ejemplo: XGBoost, LightGBM.

## C
*   **Cross-Validation (Validación Cruzada):** Técnica para evaluar modelos dividiendo la data en K partes (folds) y rotando entrenamiento/validación.

## D
*   **Data Leakage (Fuga de Datos):** Cuando información del target se filtra en las features de entrenamiento, creando modelos optimistas que fallan en producción.

## F
*   **Feature Engineering:** El arte de crear nuevas variables a partir de las existentes para mejorar el modelo.
*   **F1-Score:** Media armónica entre Precision y Recall. Útil para clases desbalanceadas.

## H
*   **Hiperparámetro:** Configuración externa del modelo que no se aprende de los datos (ej. profundidad del árbol, learning rate).

## O
*   **Optuna:** Framework de optimización de hiperparámetros que usa búsqueda bayesiana.
*   **Overfitting (Sobreajuste):** Cuando el modelo memoriza el ruido del set de entrenamiento y falla en datos nuevos.

## P
*   **Pipeline:** Cadena de pasos de procesamiento (limpieza, transformación, modelo) encapsulada en un solo objeto.
*   **Precision:** De los que predije positivos, ¿cuántos lo eran realmente? (Calidad).

## R
*   **Recall (Sensibilidad):** De todos los positivos reales, ¿cuántos detecté? (Cantidad).
*   **Regularización:** Técnica para penalizar la complejidad del modelo y evitar overfitting (L1/Lasso, L2/Ridge).

## S
*   **SHAP (SHapley Additive exPlanations):** Método basado en teoría de juegos para explicar la contribución de cada feature a una predicción.
*   **Stratified K-Fold:** Validación cruzada que mantiene la proporción de clases en cada fold.

## T
*   **Target Encoding:** Reemplazar una categoría por el promedio del target para esa categoría. Riesgo alto de leakage.

## U
*   **Underfitting:** Cuando el modelo es demasiado simple para capturar el patrón de los datos.

## V
*   **Variance (Varianza):** Sensibilidad del modelo a pequeños cambios en los datos de entrenamiento. Alta varianza -> Overfitting.

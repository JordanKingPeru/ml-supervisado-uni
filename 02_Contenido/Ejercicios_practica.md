# Ejercicios de práctica — Aprendizaje Supervisado

Ejercicios guiados por módulo, pensados para resolverse DESPUÉS de cada sesión (30–60 min cada uno).
Usan los datasets de `03_Datos/tabular/` (cárgalos con `load_data()` de `recursos/utils.py`) o datasets
de `scikit-learn`. El proyecto final integra todo el curso.

## M01 — Pipelines y baselines

1. **Baseline honesto.** Con `credit_scoring.csv`, entrena un `DummyClassifier` (estrategias
   `most_frequent` y `stratified`) y una regresión logística dentro de un `Pipeline` con imputación y
   escalado. Reporta ROC-AUC de los tres con validación cruzada estratificada (5 folds). ¿Cuánto valor
   real agrega la logística sobre el dummy?
2. **Cazafugas.** Toma tu pipeline del ejercicio 1 e introduce a propósito una fuga: ajusta el
   `StandardScaler` sobre TODO el dataset antes del split. Mide la diferencia de AUC entre la versión
   con fuga y la correcta. Explica en 3 líneas por qué la diferencia engaña.
3. **Target Encoding vs OneHot.** Reemplaza el OneHot de las variables categóricas por
   `category_encoders.TargetEncoder` (dentro del pipeline, nunca fuera). Compara AUC y tiempo de
   entrenamiento. ¿Cuándo conviene cada uno?

## M02 — Árboles y ensambles

1. **El sobreajuste visible.** Entrena un `DecisionTreeClassifier` sobre `telco_churn.csv`
   (recuerda `sep=';'`) con `max_depth` ∈ {2, 5, 10, None}. Grafica AUC en train vs validación para
   cada profundidad. ¿Dónde empieza a memorizar?
2. **Tu propia arena.** Replica la "Arena de Combate" agregando dos competidores que no estaban:
   `ExtraTreesClassifier` y `CatBoostClassifier`. Mantén el mismo `Pipeline` de preprocesamiento y la
   misma validación. Publica la tabla final ordenada por AUC y tiempo.
3. **KNN con escala y sin escala.** Demuestra con `telco_churn` que KNN sin escalado es injusto con las
   variables de mayor magnitud: compara AUC con y sin `StandardScaler` en el pipeline.

## M03 — Optimización y negocio

1. **Optuna con presupuesto.** Optimiza un LightGBM sobre `credit_scoring.csv` con Optuna limitado a
   30 trials. Reporta los 3 mejores conjuntos de hiperparámetros y el gain sobre los parámetros por
   defecto. ¿Valió la pena el costo de cómputo?
2. **Curva de ganancia.** Define una matriz de costos para el crédito (p. ej., aprobar a un mal pagador
   cuesta 5× lo que gana aprobar a uno bueno) y construye la profit curve del modelo optimizado.
   ¿En qué umbral se maximiza la ganancia? ¿Coincide con el umbral 0.5?
3. **¿Está calibrado?** Dibuja la curva de calibración (`CalibrationDisplay`) del modelo del ejercicio 1
   antes y después de `CalibratedClassifierCV`. ¿Cambió el umbral óptimo de negocio del ejercicio 2?

## M04 — Validación, XAI y producción

1. **Validación que no miente.** Sobre `DS_Compra.csv`, compara `KFold` simple vs `StratifiedKFold`
   (5 folds, misma semilla). Reporta media ± desviación del AUC. ¿Qué pasa si la clase positiva es rara?
2. **SHAP para negocio.** Calcula valores SHAP del mejor modelo de M03 y redacta — en lenguaje de
   gerencia, sin jerga — las 3 variables que más empujan el riesgo y cómo lo hacen (dirección y magnitud).
3. **Del notebook a la app.** Serializa tu modelo con `joblib` incluyendo el pipeline completo y un
   diccionario de metadatos (versión, métricas, fecha de entrenamiento generada con
   `datetime.now()`). Cárgalo en la app de `M04_XAI_Produccion/app/` y verifica que predice igual que
   en el notebook (misma fila → misma probabilidad).

## M06 (bonus) — Datos desde fuera

1. **De UCI a pipeline.** Usando `ucimlrepo` (como en `01conexión_db.ipynb`), descarga un dataset de
   clasificación de UCI distinto al del notebook y pásalo entero por tu pipeline de M01. ¿Qué tuviste
   que adaptar del preprocesamiento?

## 🎯 Proyecto final integrador

Construye, en un único notebook reproducible (usa `recursos/notebook_template.ipynb` como base), un
caso completo con `credit_scoring.csv` **o** un dataset propio de tu trabajo:

1. **EDA breve** con conclusiones accionables (no un volcado de gráficos).
2. **Baseline** (dummy + logística en pipeline) y al menos **2 retadores** (uno debe ser boosting).
3. **Optimización** con Optuna (≥20 trials) y validación estratificada.
4. **Traducción a negocio**: matriz de costos, profit curve y elección de umbral justificada.
5. **Explicabilidad**: SHAP global + 2 casos individuales explicados en lenguaje no técnico.
6. **Entregable de producción**: modelo serializado + metadatos, y celda final que lo recarga y predice.

> Criterio de oro: otra persona debe poder ejecutar tu notebook de arriba a abajo sin editar nada
> (`Kernel → Restart & Run All`).

> 📐 **Modelo de referencia:** el módulo `M07_Caso_Integrador_ML/` es un ejemplo completo y "pro" de
> este proyecto final, resuelto como **documento regulatorio** de riesgo de crédito con un dataset real
> (UCI). Úsalo como plantilla de estructura, narrativa y nivel de exigencia.

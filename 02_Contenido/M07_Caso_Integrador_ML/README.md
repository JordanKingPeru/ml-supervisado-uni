# M07 — Caso integrador end-to-end (🎁 bonus de cierre)

Documento de **validación de un modelo de riesgo de crédito** que recorre **todo el ciclo de vida**
de un modelo de ML, formateado como **entregable regulatorio** (carátula, banner confidencial, índice,
secciones numeradas). Es el cierre del curso: integra Pipelines (M01), algoritmos (M02), optimización
y negocio (M03) y XAI/producción (M04) en un solo caso realista.

## Qué contiene

- **Dataset real:** *Default of Credit Card Clients* (UCI, Taiwán) — 30.000 clientes, descargado en
  vivo con `ucimlrepo` (no simulado).
- **Dos modelos:** un **scorecard WoE + logística** (interpretable, "el que aprueba el regulador") y un
  **LightGBM challenger** optimizado con Optuna y explicado con SHAP.
- **Marco regulatorio:** terminología y lineamientos de gestión de riesgo de crédito (SBS, Perú): IV/WoE,
  scorecard, KS/Gini, calibración, **PSI/estabilidad**, backtesting, provisiones.
- **12 secciones:** contexto y datos → EDA → target → matriz de variables (IV/WoE) → entrenamiento →
  sesgo/varianza → hiperparámetros → desempeño → estabilidad temporal (PSI) → impacto de negocio →
  conclusiones y anexos.

## Archivos

| Archivo | Qué es |
|---|---|
| `notebooks/Reporte_Modelo_Riesgo_Credito.ipynb` | El documento (ejecutable de arriba a abajo). |
| `notebooks/build_reporte.py` | Generador del notebook (fuente editable; re-ejecutar para regenerar). |
| `notebooks/estilos.css` | Estilos del HTML regulatorio (carátula, banner, tablas). |
| `salidas/Reporte_Modelo_Riesgo_Credito.html` | Documento exportado a HTML profesional. |
| `salidas/Reporte_Modelo_Riesgo_Credito.pdf` | Versión PDF (para el regulador / comité). |

## Cómo ejecutarlo

```bash
pip install -r ../M00_Onboarding/requirements.txt   # incluye optbinning
jupyter lab   # abrir Reporte_Modelo_Riesgo_Credito.ipynb y Restart & Run All
```

Un único bloque de configuración fija la paleta y el tema, de modo que **todos los gráficos comparten
la misma estética**. Requiere conexión a internet la primera vez (descarga del dataset UCI).

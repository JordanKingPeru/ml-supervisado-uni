# M08 — Catálogo de Algoritmos (🎁 referencia transversal)

Deck de **referencia rápida** con ~30 algoritmos de aprendizaje supervisado, de la regresión lineal a
los *foundation models* y los pre-entrenados de Google. **No es una sesión de dictado**: es material de
consulta que acompaña a todo el curso.

## Qué contiene

- **Mapa de familias** (taxonomía visual) y **tabla maestra** comparativa de los ~30 algoritmos
  (interpretabilidad, no-linealidad, ¿escalar?, tamaño de datos, velocidad, librería, nivel).
- **Panel comparativo de fronteras de decisión**: el mismo problema resuelto por 12 algoritmos.
- **Guía de decisión** (¿qué uso?) + receta práctica.
- **1 lámina por algoritmo** con ficha consistente: 🧠 intuición + frontera · ✅ pros / ❌ contras ·
  🎯 cuándo usar / 🚫 evitar · ⚠️ condiciones · 💻 librería+código · 📚 docs · 🧭 ejemplo en el curso ·
  📊 ficha técnica (complejidad, creador/año, requisitos).

## Cobertura (multi-librería, SOTA y pre-entrenados)

Lineales/GLM · KNN · Naive Bayes · LDA/QDA · SVM · Árbol · Random Forest · Extra Trees · AdaBoost ·
Gradient Boosting · HistGradientBoosting · **XGBoost · LightGBM · CatBoost** · NGBoost · Voting/Stacking ·
MLP · **TabNet** · FT-Transformer/SAINT · **TabPFN** (foundation) · **AutoML** (AutoGluon/FLAML) ·
**Google**: TF Decision Forests, Vertex AI, **Gemini zero-shot**, Embeddings, Hugging Face.

## Archivos

| Archivo | Qué es |
|---|---|
| `slides/S08_Presentacion.md` | El deck (fuente Marp). |
| `slides/S08_Presentacion.pdf` | El catálogo listo para usar/imprimir. |
| `slides/assets/build_figuras.py` | Genera las fronteras de decisión (reproducible). |
| `slides/assets/*.svg` | Fronteras + diagramas (mapa de familias, guía de decisión). |

> Cada tarjeta enlaza al notebook de ejemplo del algoritmo en `02_Contenido/Mxx_*/notebooks/`.

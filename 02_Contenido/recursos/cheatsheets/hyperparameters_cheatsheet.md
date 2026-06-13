# 🌳 Cheatsheet: Hiperparámetros de Árboles (XGBoost / LightGBM / RF)

## 🌲 Random Forest (Bagging)
*Objetivo: Reducir Varianza (Overfitting)*

| Parámetro (sklearn) | Significado | Efecto al Aumentar | Valor Típico |
|---|---|---|---|
| `n_estimators` | Número de árboles | Mejor estabilidad, más lento. No causa overfitting. | 100 - 1000 |
| `max_depth` | Profundidad máxima | Más complejidad, riesgo de overfitting. | 10 - None |
| `min_samples_leaf` | Mínimo de muestras por hoja | Suaviza el modelo (Regularización). | 1 - 50 |
| `max_features` | Features por split | Menor valor = Árboles más diversos (menos correlacionados). | 'sqrt' o 'log2' |

---

## 🚀 XGBoost / LightGBM (Boosting)
*Objetivo: Reducir Bias y Varianza*

| Parámetro (LGBM / XGB) | Significado | Efecto | Rango Típico |
|---|---|---|---|
| `learning_rate` (eta) | Velocidad de aprendizaje | **Bajo:** Más robusto, requiere más árboles. **Alto:** Rápido, riesgo overfitting. | 0.01 - 0.3 |
| `n_estimators` (num_boost_round) | Número de iteraciones | Debe ajustarse con `learning_rate`. (LR bajo -> N alto). | 100 - 5000 (con early stopping) |
| `num_leaves` (max_leaves) | Complejidad del árbol | Control principal de complejidad en LGBM. | 20 - 100 |
| `max_depth` | Profundidad máxima | Limita `num_leaves`. Útil para evitar overfitting extremo. | 3 - 12 |
| `subsample` (bagging_fraction) | % Datos por árbol | Previene overfitting. Acelera entrenamiento. | 0.5 - 0.9 |
| `colsample_bytree` (feature_fraction) | % Features por árbol | Previene overfitting. | 0.5 - 0.9 |
| `scale_pos_weight` | Peso clase positiva | **CRÍTICO** para desbalance. | sum(neg) / sum(pos) |

---

## ⚖️ Guía de Tuning Rápida

1.  **Overfitting (Train >>> Test):**
    *   ⬇️ `max_depth` / `num_leaves`
    *   ⬆️ `min_samples_leaf` / `min_child_weight`
    *   ⬇️ `subsample` / `colsample_bytree`
    *   ⬆️ `reg_alpha` (L1) / `reg_lambda` (L2)

2.  **Underfitting (Train bajo y Test bajo):**
    *   ⬆️ `max_depth` / `num_leaves`
    *   ⬇️ `learning_rate` (y aumentar `n_estimators`)
    *   ⬇️ Regularización

3.  **Velocidad:**
    *   Usar `LightGBM` en lugar de XGBoost.
    *   Usar GPU (`device='gpu'`).
    *   Aumentar `batch_size` (si aplica).

# 📄 Cheatsheet: Scikit-Learn Pipelines & Preprocessing

## 🛠️ Estructura Básica de un Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# 1. Definir pasos para numéricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 2. Definir pasos para categóricas
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 3. Combinar en ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Pipeline Final con Modelo
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])
```

## 🔄 Transformadores Comunes

| Transformador | Uso Principal | Cuándo usarlo |
|---|---|---|
| `SimpleImputer` | Rellenar nulos | Datos faltantes básicos (media, mediana, constante). |
| `KNNImputer` | Rellenar nulos | Datos faltantes complejos (usa vecinos cercanos). Más lento. |
| `StandardScaler` | Escalar (Media 0, Std 1) | Modelos lineales (Ridge, Lasso, Logistic), SVM, Neural Nets. |
| `MinMaxScaler` | Escalar (0 a 1) | Algoritmos que requieren distancias acotadas o imágenes. |
| `RobustScaler` | Escalar (Mediana, IQR) | Cuando hay **Outliers** extremos. |
| `OneHotEncoder` | Categoría a Columnas | Categorías nominales (sin orden: Color, Ciudad). |
| `OrdinalEncoder` | Categoría a Enteros | Categorías ordinales (con orden: Bajo, Medio, Alto). |
| `TargetEncoder` | Categoría a Promedio Target | Categorías con alta cardinalidad (muchos valores únicos). ¡Cuidado con Leakage! |

## ⚠️ Errores Comunes (Anti-Patterns)

1.  **Fit en todo el dataset:** NUNCA hagas `fit()` antes de dividir en Train/Test. El `fit` solo va en Train.
2.  **Leakage en Imputación:** Si usas la media de todo el dataset para imputar, estás filtrando información. El Pipeline maneja esto automáticamente.
3.  **Orden Incorrecto:** Primero `Imputer`, luego `Scaler`. Si escalas antes de imputar, los nulos pueden romper el scaler o el valor de relleno quedará sin escalar.

## 💡 Pro-Tip: Visualización
Usa `sklearn.set_config(display="diagram")` para ver tu pipeline como un diagrama interactivo en Jupyter.

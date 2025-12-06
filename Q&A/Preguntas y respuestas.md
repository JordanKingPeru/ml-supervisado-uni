# Respuestas ampliadas a preguntas seleccionadas del curso

Este documento recoge **respuestas desarrolladas** a algunas de las preguntas que surgieron en clase.  
La idea es que funcione como material de lectura y referencia para que puedan volver sobre los conceptos con calma.

Las preguntas abordadas aquí son:

- P4. ¿Se va a abordar el coeficiente de Gini en la teoría de la regresión logística? ¿Qué es y por qué se usa tanto en *credit scoring*?
- P5. ¿Qué son los efectos marginales y cómo se interpretan en regresión logística?
- P6. ¿Se puede decir que el monto del pago es decisivo para ver si el cliente se va o no (fuga/churn)?
- P7. ¿Cómo podemos calcular el precio máximo sin riesgo de fuga (optimización)?
- P10. ¿Qué determina que una variable sea adecuada para un modelo correcto?

---

## P4. Coeficiente de Gini en regresión logística y *credit scoring*

**Pregunta:**  
> ¿Se va a abordar el coeficiente de Gini en la teoría de la regresión logística? ¿Qué es y por qué se usa tanto en *credit scoring*?

### 1. ¿Qué es el coeficiente de Gini en el contexto de modelos?

En modelos de clasificación binaria (p.e. aprobar/rechazar, pagar/no pagar, churn/no churn), el **coeficiente de Gini** se usa como medida de **capacidad discriminante**:  
> Qué tan bien separa el modelo a los “buenos” de los “malos”.

Está íntimamente relacionado con el **AUC (Area Under the ROC Curve)**:

\[
\text{Gini} = 2 \times \text{AUC} - 1
\]

- Si AUC = 0.5 (modelo aleatorio) → Gini = 0  
- Si AUC = 1 (modelo perfecto) → Gini = 1  
- Gini puede teóricamente ir de -1 a 1, pero en la práctica, valores negativos indican un modelo “al revés”.

En *credit scoring*, suele reportarse **Gini %**, por ejemplo:  
- Gini = 0.4 → 40%  
- Gini = 0.6 → 60%

### 2. Por qué se usa tanto en *credit scoring*

En *credit scoring* (modelos de riesgo de crédito):

- El objetivo central es **ordenar** clientes por riesgo: quién es más probable que caiga en mora.  
- El negocio muchas veces toma decisiones con **umbrales sobre ese ranking** (top 10% más riesgoso, etc.).  

El Gini:

1. Resume en **un solo número** la capacidad de discriminación del modelo.  
2. Es **independiente del umbral** (a diferencia de la exactitud, que depende del “corte” elegido).  
3. Se interpreta como cuán mejor es el modelo que el azar en términos de ranking de riesgo.

Por eso, en bancos y financieras es muy común escuchar frases como:  
> “Este modelo de originación tiene Gini 55%”,  
como indicador estándar de calidad.

### 3. Sensibilidad y características importantes del Gini

#### (a) Sensibilidad a cambios en las posiciones del *ranking*

El Gini (como el AUC) solo se fija en el **orden relativo** de los scores, no en los valores absolutos.

- Si reescalas el score con una función **monótona creciente** (ej. log, exp, suma de constante), el Gini **no cambia**, porque el orden se mantiene.
- Pequeños cambios que alteran el **orden** (especialmente en la parte alta del riesgo) pueden afectar el Gini de forma apreciable.  

Esto implica:

- El Gini es **sensible a errores de ordenamiento**, sobre todo para los casos positivos (por ejemplo, morosos o clientes con default).

#### (b) Sensibilidad al tamaño de muestra y a la proporción de “eventos”

- Con **muy pocos eventos** (pocas observaciones de “malos”), el Gini puede ser **inestable**: un solo caso mal ordenado puede cambiar mucho el resultado.  
- En muestras pequeñas, es prudente:
  - Usar **intervalos de confianza** (bootstrap) para el Gini.
  - No interpretar diferencias pequeñas (ej. 0.52 vs 0.54) como “ganancias enormes”.

La **prevalencia** (porcentaje de malos) no afecta directamente el Gini como sí afecta otras métricas (ej. exactitud), pero sí influye en la **precisión de la estimación**.

#### (c) Robustez ante cambios de escala

Propiedades clave:

- **Invariante a transformaciones monótonas** del score → asegura que si cambias de “score bruto” a “probabilidad” o a “puntos de score” (como en scoring de crédito), el Gini se mantiene, mientras el orden se conserve.
- Depende solo del **ranking**, lo que lo hace robusto ante cambios de calibración (p.e. si todas las probabilidades están un poco subestimadas o sobreestimadas, pero el orden es correcto).

#### (d) Limitaciones y malas interpretaciones comunes

- El Gini **no mide calibración**:  
  - Un modelo puede tener Gini alto (ordenar bien) pero **probabilidades mal calibradas** (p.e. predecir 80% cuando la prob real es 40%).  
- Comparar Gini entre **poblaciones muy distintas** (países, segmentos) puede ser engañoso.  
- Gini alto no garantiza que el modelo sea **justo o ético**: solo habla de discriminación estadística, no de sesgos.

En resumen, el Gini es excelente para evaluar **discriminación / ranking**, pero debe complementarse con otras métricas y análisis.

---

## P5. Efectos marginales en regresión logística

**Pregunta:**  
> ¿Qué son los efectos marginales y cómo se interpretan en regresión logística?

### 1. Problema: coeficientes en “log-odds”

En regresión logística, el modelo es:

\\[
P(Y=1 \mid X) = p(X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \cdots + \beta_k X_k)}}
\\]

Los coeficientes \(\beta_j\) están en términos de **log-odds**:

\[
\log\left(\frac{p(X)}{1-p(X)}\right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_k X_k
\]

Interpretar “1 unidad en \(X_j\) cambia el log-odds en \(\beta_j\)” no es muy intuitivo para un usuario de negocio.

### 2. ¿Qué es el efecto marginal?

El **efecto marginal** de una variable continua \(X_j\) sobre la probabilidad es:

\[
\frac{\partial p(X)}{\partial X_j} = \beta_j \cdot p(X) \cdot (1 - p(X))
\]

Interpretación:

> Es el cambio aproximado en la **probabilidad** cuando \(X_j\) aumenta en una unidad, manteniendo las demás variables constantes.

- Si el efecto marginal de `Ingreso` es 0.02 → una unidad más de ingreso incrementa la probabilidad en **2 puntos porcentuales** (aprox.) en ese punto.

### 3. Efectos marginales promedio (AME) vs. en un punto (MEM)

En la práctica se usan dos tipos:

1. **Marginal Effect at the Mean (MEM)**:  
   - Se evalúa el efecto en un **punto típico** (por ejemplo, en la media de las variables).  
   - “Si el cliente es ‘promedio’, aumentar X en 1 unidad cambia la probabilidad en Δ”.

2. **Average Marginal Effect (AME)**:  
   - Se calcula el efecto marginal para **cada observación** y luego se promedia.  
   - Más representativo de “la población”, pero menos intuitivo de visualizar en una sola persona.

### 4. Efectos marginales para variables binarias

Si \(X_j\) es binaria (0/1), el efecto marginal se interpreta como:

\[
\Delta p = p(Y=1 \mid X_j=1, \text{otros}) - p(Y=1 \mid X_j=0, \text{otros})
\]

Ejemplo:

- Variable `Tiene_tarjeta` (0 = no, 1 = sí).  
- Efecto marginal = 0.15 →  
  > Tener tarjeta aumenta la probabilidad de churn en 15 puntos porcentuales, manteniendo fijo lo demás.

### 5. Por qué son útiles

- Traducen el modelo a **lenguaje de probabilidad**, mucho más amigable para negocio.  
- Permiten comparar **relevancia práctica** de variables:  
  - Una variable puede tener un \(\beta_j\) grande pero un impacto marginal pequeño si se mueve en un rango reducido o si la probabilidad base es muy alta/baja.

---

## P6. ¿Es el monto del pago decisivo en churn?

**Pregunta:**  
> ¿Se puede decir que el monto del pago es decisivo para ver si el cliente se va o no (fuga/churn)?

### 1. Respuesta corta

Depende de los datos y del modelo, pero **rara vez hay una sola variable “decisiva”**.  
El monto del pago puede ser una variable muy importante, pero **no actúa en el vacío**.

### 2. Cómo evaluar la “importancia” de una variable

Para saber si el pago es clave, podemos usar:

- En **regresión logística**:
  - Significancia del coeficiente (p-valor).
  - Magnitud del efecto marginal.
- En **árboles / random forests / gradient boosting**:
  - Importancia de variables (*feature importance*).
  - Frecuencia con que la variable aparece en splits altos del árbol.
- Análisis avanzados:
  - **SHAP values** o **partial dependence plots** para ver el efecto del pago manteniendo otras variables promedio.

Si, por ejemplo:

- El pago aparece siempre en los **primeros splits** del árbol.  
- El **efecto marginal** es grande.  
- Y al quitar esa variable el Gini/AUC cae bastante.  

Entonces podemos decir que el monto del pago es **muy influyente** en la predicción de churn.

### 3. Cuidado: correlación vs. causalidad

Que el monto del pago sea muy importante **no significa automáticamente** que bajando el pago se evitará el churn, porque:

- Puede estar correlacionado con otras cosas (ingreso, tipo de segmento, canal de venta…).  
- Puede reflejar decisiones anteriores (el cliente ya estaba en riesgo y se le ofreció un descuento).

Para tomar decisiones (por ejemplo, campañas de descuento) conviene:

- Analizar resultados de **experimentos A/B** o de campañas pasadas.  
- Combinar el modelo con **criterio de negocio**.

---

## P7. Cálculo del “precio máximo sin riesgo de fuga”

**Pregunta:**  
> ¿Cómo podemos calcular el precio máximo sin riesgo de fuga (optimización)?

Primero: **no existe un precio con “cero” riesgo de fuga**.  
Siempre habrá alguna probabilidad de que el cliente se vaya.  
Lo que sí podemos hacer es buscar el **precio que maximiza el beneficio esperado**, teniendo en cuenta el riesgo.

### 1. Idea general

Supongamos que tenemos:

- Un modelo que estima \(P(\text{fuga} \mid \text{precio}, \text{otras variables})\).  
- Un coste `c` y un precio `p`.  
- Si el cliente **permanece**, ganamos aproximadamente `(p − c)` (margen).  
- Si el cliente **se va**, la contribución de ese período es 0 (o incluso negativa si hay costes fijos).

Entonces el **beneficio esperado** para un cliente a un precio `p` puede ser:

\[
\text{Beneficio esperado}(p) \approx (p - c) \times (1 - P(\text{fuga} \mid p))
\]

> Elegimos el `p` que maximiza esta función.

### 2. Pasos prácticos

1. Estimar un modelo de churn donde el **precio (o descuento)** sea una variable explicativa.  
2. Definir la función de **beneficio** (o margen) por cliente según el precio.  
3. Para un conjunto de precios candidatos (p.e. `p` en un rango razonable):
   - Calcular \(P(\text{fuga} \mid p)\) con el modelo.  
   - Calcular beneficio esperado \((p - c)(1 - P(\text{fuga} \mid p))\).
4. Elegir el precio que dé **mayor beneficio esperado**, no necesariamente el de menor churn.

### 3. Extensiones

- Incluir valor de **largo plazo**:  
  - Valor presente neto de los flujos futuros si el cliente se queda.  
- Incorporar restricciones:  
  - Límites regulatorios o comerciales.  
  - Segmentos donde el objetivo sea **retención** más que margen inmediato.

De nuevo: no buscamos “precio sin riesgo”, sino el mejor compromiso entre **precio alto** y **probabilidad de retener al cliente**.

---

## P10. ¿Qué hace que una variable sea adecuada para un modelo?

**Pregunta:**  
> ¿Qué determina que cada variable sea adecuada para un modelo correcto?

### 1. Perspectiva estadística

Desde el punto de vista técnico, una variable es buena candidata si:

1. Tiene **relación informativa** con la variable objetivo.  
2. No está completamente llena de **valores perdidos**.  
3. No es una copia casi idéntica de otras variables (colinealidad severa en regresión).  
4. Es **estable en el tiempo** (distribución similar entre entrenamiento y producción).

Herramientas típicas:

- Correlación, gráficos (boxplots, scatterplots, binning…).  
- En *credit scoring*, indicadores como **Weight of Evidence (WoE)** e **Information Value (IV)**.  
- **PSI (Population Stability Index)** para ver si la variable cambia de comportamiento en el tiempo.

### 2. Perspectiva de negocio

Desde el negocio, una variable debería:

- Tener **sentido lógico** (que el efecto que sugiere el modelo no sea absurdo).  
- No generar **conflictos éticos o regulatorios** (ej. variables prohibidas como raza, religión, etc.).  
- Ser **accionable** o, al menos, útil para comprender el comportamiento del cliente.

### 3. Perspectiva de implementación

Además, hay criterios prácticos:

- ¿La variable está disponible **a tiempo** cuando se necesita la predicción?  
  - Ej.: una variable que solo se conoce un mes después no sirve para decisiones en tiempo real.  
- ¿Se puede mantener su calidad **de forma operativa** (sin errores masivos, sin cambios constantes de definición)?  
- ¿Es una variable “futura” que introduce **data leakage**?  
  - Ej.: “días en mora en los próximos 90 días” no puede usarse para un modelo que decide hoy.

### 4. Resumen de checklist para una “buena variable”

Para decidir si una variable se queda en el modelo, preguntarse:

1. **¿Aporta información?**  
   - Mejora Gini/AUC, reduce error, aumenta log-likelihood, etc.  
2. **¿Es estable y de calidad?**  
   - Pocos valores faltantes, sin cambios bruscos de distribución, definición clara.  
3. **¿Es aceptable para regulador y negocio?**  
   - Sin discriminar ilegalmente, con lógica de negocio.  
4. **¿Es usable en producción?**  
   - Disponible a tiempo, calculable de forma confiable.

Si una variable falla en varios de estos puntos, aunque sea “predictiva” en una muestra, probablemente **no es adecuada** para un modelo que vaya a producción.

---

## Notas finales

Este documento está pensado como material de lectura para reforzar:

- Métricas como el **coeficiente de Gini** y su interpretación.  
- Conceptos de **regresión logística**, como efectos marginales.  
- Criterios de negocio y estadística para el **uso de variables** y la **optimización de precios**.

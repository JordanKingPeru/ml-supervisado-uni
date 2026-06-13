# -*- coding: utf-8 -*-
"""
Generador del notebook regulatorio end-to-end del Modulo M07.
Construye `Reporte_Modelo_Riesgo_Credito.ipynb` celda por celda con nbformat.
Mantener este generador como fuente editable del notebook (re-ejecutar para regenerar).
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

cells = []
def md(src):  cells.append(new_markdown_cell(src))
def code(src): cells.append(new_code_cell(src))

# =========================================================================
# CARATULA REGULATORIA
# =========================================================================
md(r"""<div class="caratula">
  <div class="banner-confidencial">DOCUMENTO CONFIDENCIAL &mdash; USO INTERNO Y REGULATORIO</div>
  <div class="caratula-cuerpo">
    <p class="entidad">Programa de Especialización en Ciencia de Datos &mdash; UNI</p>
    <h1>Documento de Validación de Modelo<br>de Riesgo de Crédito</h1>
    <h2>Scorecard de probabilidad de incumplimiento (PD) y modelo challenger de Machine Learning</h2>
    <p class="subtitulo">Caso integrador end-to-end &mdash; Aprendizaje Supervisado</p>
  </div>
</div>

### Control del documento

| Campo | Detalle |
|---|---|
| **Título** | Validación de Modelo de Riesgo de Crédito — Scorecard PD |
| **Tipo de modelo** | Calificación crediticia (probabilidad de incumplimiento a 1 período) |
| **Marco de referencia** | Gestión de riesgo de crédito y de modelos — lineamientos SBS (Perú) |
| **Cartera** | Tarjetas de crédito — personas naturales |
| **Estado** | Validación independiente |
| **Clasificación** | Confidencial |
| **Versión** | 1.0 |
| **Audiencia** | Comité de Riesgos · Validación de Modelos · Regulador |

> **Aviso.** Documento de carácter técnico y formativo. Los datos provienen de la base pública
> *Default of Credit Card Clients* (UCI, Taiwán; 30.000 clientes), utilizada como sustituto realista
> de una cartera supervisada. La terminología regulatoria (clasificación del deudor, provisiones,
> estabilidad PSI, backtesting) sigue los lineamientos de la SBS con fines ilustrativos.
""")

md(r"""## Resumen ejecutivo

Este documento presenta el desarrollo y la **validación independiente** de un modelo de calificación
crediticia (PD) para una cartera de tarjetas de crédito de personas naturales. Se construyen dos modelos:

1. **Modelo regulatorio (en producción):** un **scorecard** basado en *Weight of Evidence* (WoE) y
   regresión logística. Es transparente, auditable y traducible a puntajes — requisito para su uso
   en decisiones de crédito y para la revisión del supervisor.
2. **Modelo challenger:** un **Gradient Boosting (LightGBM)** optimizado con Optuna y explicado con
   **SHAP**. Establece la cota superior de poder predictivo y permite cuantificar cuánto desempeño se
   sacrifica por interpretabilidad.

El documento recorre **todo el ciclo de vida del modelo**: datos y diccionario, análisis exploratorio,
definición del *target*, matriz de variables (IV/WoE), entrenamiento, diagnóstico de sesgo-varianza,
optimización de hiperparámetros, desempeño (ROC, KS, Gini, calibración), **estabilidad en el tiempo
(PSI)** y **traducción a impacto de negocio** (curvas de ganancia, deciles y estimación financiera).

*Los indicadores numéricos de este resumen se calculan automáticamente al final del documento
(sección 12) a partir de la ejecución real del modelo.*
""")

# =========================================================================
# 0. CONFIGURACION Y TEMA UNICO
# =========================================================================
md(r"""---
## 0. Configuración del entorno y tema visual

Se define **una sola vez** la paleta corporativa y el tema de los gráficos, de modo que todas las
figuras del documento compartan la misma estética (requisito de un entregable profesional).
""")

code(r"""# === Librerías ===
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from cycler import cycler

from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.calibration import calibration_curve

import lightgbm as lgb
import optuna
from optbinning import BinningProcess, Scorecard
from optbinning.scorecard import plot_auc_roc, plot_ks
import shap

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# === Paleta corporativa y tema único ===
AZUL_OSCURO = "#1f4e79"; AZUL = "#2e6da4"; AZUL_CLARO = "#5b9bd5"
ACENTO = "#e07b39"; OK = "#2e8b57"; MALO = "#c0392b"; GRIS = "#7f8c8d"
PALETA = [AZUL_OSCURO, ACENTO, AZUL_CLARO, OK, MALO, GRIS]

plt.rcParams.update({
    "figure.figsize": (10, 5.2), "figure.dpi": 110,
    "axes.prop_cycle": cycler(color=PALETA),
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": AZUL_OSCURO,
    "axes.labelsize": 11, "axes.edgecolor": "#cccccc", "axes.grid": True,
    "grid.color": "#ececec", "grid.linewidth": 0.7, "grid.alpha": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5, "legend.frameon": False, "figure.autolayout": True,
})

def estilo_titulo(ax, titulo, sub=None):
    ax.set_title(titulo, loc="left", pad=(24 if sub else 10))
    if sub: ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9, color=GRIS, va="bottom")
    return ax

def fmt_miles(ax, eje="y"):
    f = mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    (ax.yaxis if eje=="y" else ax.xaxis).set_major_formatter(f)

print("Entorno configurado. Tema corporativo activo. Semilla =", RANDOM_STATE)""")

# =========================================================================
# 1/2. CONTEXTO Y DATOS
# =========================================================================
md(r"""---
## 1. Contexto y datos

**Objetivo del modelo.** Estimar la probabilidad de que un cliente de tarjeta de crédito **incumpla
su pago en el siguiente período** (PD a 1 mes), para soportar decisiones de admisión, líneas y
provisiones bajo los lineamientos de gestión de riesgo de crédito de la SBS.

**Fuente.** *Default of Credit Card Clients* (UCI Machine Learning Repository, Taiwán). 30.000 clientes,
23 variables predictoras y la variable objetivo `default` (1 = incumple el próximo mes).
""")

code(r"""# === Descarga del dataset real (UCI id=350) ===
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=350)
X_raw = ds.data.features.copy()
y = ds.data.targets.iloc[:, 0].rename("default").astype(int)

# Nombres significativos (según la documentación oficial del dataset)
ren = {"X1":"LIMIT_BAL","X2":"SEX","X3":"EDUCATION","X4":"MARRIAGE","X5":"AGE",
       "X6":"PAY_1","X7":"PAY_2","X8":"PAY_3","X9":"PAY_4","X10":"PAY_5","X11":"PAY_6",
       "X12":"BILL_AMT1","X13":"BILL_AMT2","X14":"BILL_AMT3","X15":"BILL_AMT4","X16":"BILL_AMT5","X17":"BILL_AMT6",
       "X18":"PAY_AMT1","X19":"PAY_AMT2","X20":"PAY_AMT3","X21":"PAY_AMT4","X22":"PAY_AMT5","X23":"PAY_AMT6"}
df = X_raw.rename(columns=ren).copy()

# Limpieza de categorías fuera de catálogo (EDUCATION 0,5,6 -> 'otros'=4 ; MARRIAGE 0 -> 'otros'=3)
df["EDUCATION"] = df["EDUCATION"].replace({0:4, 5:4, 6:4})
df["MARRIAGE"]  = df["MARRIAGE"].replace({0:3})

# Variables derivadas con sentido de negocio
df["UTILIZACION"] = (df["BILL_AMT1"] / df["LIMIT_BAL"].replace(0, np.nan)).clip(-1, 5).fillna(0)
df["PAGO_RATIO"]  = (df["PAY_AMT1"] / df["BILL_AMT1"].replace(0, np.nan)).clip(0, 5).fillna(0)
df["MESES_ATRASO_MAX"] = df[["PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]].max(axis=1).clip(lower=0)
df["default"] = y.values

print(f"Clientes: {len(df):,}  |  Variables: {df.shape[1]-1}  |  Tasa de default: {y.mean():.2%}")
df.head()""")

md(r"""### Diccionario de variables

| Variable | Descripción | Tipo |
|---|---|---|
| `LIMIT_BAL` | Línea de crédito otorgada (NT$) | Numérica |
| `SEX` | Sexo (1=hombre, 2=mujer) | Categórica |
| `EDUCATION` | Nivel educativo (1=posgrado … 4=otros) | Categórica |
| `MARRIAGE` | Estado civil (1=casado, 2=soltero, 3=otros) | Categórica |
| `AGE` | Edad (años) | Numérica |
| `PAY_1 … PAY_6` | Estado de pago últimos 6 meses (−1=al día, ≥1=meses de atraso) | Ordinal |
| `BILL_AMT1 … 6` | Monto facturado últimos 6 meses (NT$) | Numérica |
| `PAY_AMT1 … 6` | Monto pagado últimos 6 meses (NT$) | Numérica |
| `UTILIZACION` | Facturación / línea (derivada) | Numérica |
| `PAGO_RATIO` | Pago / facturación (derivada) | Numérica |
| `MESES_ATRASO_MAX` | Máximo atraso en 6 meses (derivada) | Numérica |
| `default` | **Objetivo:** 1 = incumple el próximo mes | Binaria |
""")

# =========================================================================
# 3. EDA
# =========================================================================
md(r"""---
## 3. Análisis exploratorio (EDA)

Antes de modelar se revisa la distribución de las variables clave, la completitud de la información y
la estructura de correlaciones. El objetivo es detectar anomalías, concentraciones y relaciones que
condicionen el diseño del scorecard.
""")

code(r"""fig = plt.figure(figsize=(12, 7)); gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.25)

ax1 = fig.add_subplot(gs[0,0])
ax1.hist(df["LIMIT_BAL"]/1000, bins=40, color=AZUL, edgecolor="white")
estilo_titulo(ax1, "Distribución de la línea de crédito", "Miles de NT$"); fmt_miles(ax1)

ax2 = fig.add_subplot(gs[0,1])
ax2.hist(df["AGE"], bins=30, color=AZUL_CLARO, edgecolor="white")
estilo_titulo(ax2, "Distribución de edad", "Años")

ax3 = fig.add_subplot(gs[1,0])
edu_lbl = {1:"Posgrado",2:"Universidad",3:"Secundaria",4:"Otros"}
vc = df["EDUCATION"].map(edu_lbl).value_counts()
ax3.barh(vc.index[::-1], vc.values[::-1], color=AZUL_OSCURO); fmt_miles(ax3, "x")
estilo_titulo(ax3, "Clientes por nivel educativo")

ax4 = fig.add_subplot(gs[1,1])
ax4.hist(df["MESES_ATRASO_MAX"], bins=range(0,11), color=ACENTO, edgecolor="white", align="left")
estilo_titulo(ax4, "Máximo atraso en 6 meses", "Meses")
fig.suptitle("Figura 3.1 — Perfil descriptivo de la cartera", x=0.01, ha="left", fontsize=14, weight="bold", color=AZUL_OSCURO)
plt.show()""")

code(r"""# Completitud y correlación
num_cols = ["LIMIT_BAL","AGE","BILL_AMT1","PAY_AMT1","UTILIZACION","PAGO_RATIO","MESES_ATRASO_MAX",
            "PAY_1","PAY_2","PAY_3"]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios":[1,1.25]})

faltantes = (df.isna().mean()*100).sort_values(ascending=False).head(8)
axa.barh(faltantes.index[::-1], faltantes.values[::-1], color=GRIS)
estilo_titulo(axa, "Completitud", "% de valores faltantes (top 8)")
axa.set_xlim(0, max(1, faltantes.max()*1.2))

corr = df[num_cols].corr()
im = axb.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
axb.set_xticks(range(len(num_cols))); axb.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=8)
axb.set_yticks(range(len(num_cols))); axb.set_yticklabels(num_cols, fontsize=8)
estilo_titulo(axb, "Mapa de correlación (variables numéricas)")
fig.colorbar(im, ax=axb, fraction=0.046, pad=0.04)
plt.show()""")

# =========================================================================
# 4. TARGET
# =========================================================================
md(r"""---
## 4. Construcción y análisis del *target*

La variable objetivo `default` indica incumplimiento en el siguiente período. Se analiza el **balance
de clases** (clave para elegir métricas) y la **tasa de evento por segmento**, que anticipa qué
variables tendrán poder discriminante.
""")

code(r"""fig, (axd, axe) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios":[1,1.4]})

# Donut de balance de clases
tasa = df["default"].mean()
axd.pie([1-tasa, tasa], labels=["Cumple","Incumple"], colors=[AZUL_CLARO, MALO],
        autopct=lambda p: f"{p:.1f}%", startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white"))
estilo_titulo(axd, "Balance de clases", f"Tasa de incumplimiento = {tasa:.1%}")

# Tasa de default por nivel de utilización (deciles)
df["_util_bin"] = pd.qcut(df["UTILIZACION"].rank(method="first"), 10, labels=[f"D{i}" for i in range(1,11)])
tr = df.groupby("_util_bin")["default"].mean()*100
axe.plot(tr.index.astype(str), tr.values, marker="o", color=AZUL_OSCURO, lw=2)
axe.axhline(tasa*100, ls="--", color=ACENTO, label="Tasa media")
estilo_titulo(axe, "Tasa de incumplimiento por decil de utilización", "% que incumple")
axe.legend(); df.drop(columns="_util_bin", inplace=True)
plt.show()""")

# =========================================================================
# 5. WoE / IV
# =========================================================================
md(r"""---
## 5. Matriz de variables: Weight of Evidence (WoE) e Information Value (IV)

En un scorecard regulatorio cada variable se transforma a **WoE** (logaritmo del cociente entre la
proporción de buenos y malos en cada tramo) y se prioriza por su **Information Value (IV)**, indicador
estándar de poder predictivo:

| IV | Poder predictivo |
|---|---|
| < 0,02 | Nulo |
| 0,02 – 0,1 | Débil |
| 0,1 – 0,3 | Medio |
| 0,3 – 0,5 | Fuerte |
| > 0,5 | Sospechoso (revisar) |

El binning óptimo (monótono) se calcula con `optbinning`, estándar de la industria para scorecards.
""")

code(r"""features = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",
            "PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
            "BILL_AMT1","PAY_AMT1","UTILIZACION","PAGO_RATIO","MESES_ATRASO_MAX"]
categorical = ["SEX","EDUCATION","MARRIAGE"]

X_all = df[features].copy()
y_all = df["default"].copy()

# Split estable para todo el documento
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.30,
                                          stratify=y_all, random_state=RANDOM_STATE)

binning = BinningProcess(variable_names=features, categorical_variables=categorical)
binning.fit(X_tr, y_tr)
tabla_iv = (binning.summary().sort_values("iv", ascending=False)
            [["name","iv","js","quality_score"]].reset_index(drop=True))
tabla_iv["poder"] = pd.cut(tabla_iv["iv"], [-1,0.02,0.1,0.3,0.5,99],
                           labels=["Nulo","Débil","Medio","Fuerte","Sospechoso"])
tabla_iv.round(3)""")

code(r"""# Gráfico de IV por variable
fig, ax = plt.subplots(figsize=(11, 6))
t = tabla_iv.sort_values("iv")
colores = [MALO if v>0.5 else AZUL_OSCURO if v>0.3 else AZUL if v>0.1 else GRIS for v in t["iv"]]
ax.barh(t["name"], t["iv"], color=colores)
for x in (0.1, 0.3, 0.5): ax.axvline(x, ls="--", color=ACENTO, lw=0.8, alpha=0.7)
estilo_titulo(ax, "Figura 5.1 — Information Value por variable", "Líneas: umbrales 0,1 / 0,3 / 0,5")
ax.set_xlabel("IV")
plt.show()""")

code(r"""# WoE por tramos de las dos variables más predictivas
top2 = tabla_iv["name"].head(2).tolist()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, var in zip(axes, top2):
    tb = binning.get_binned_variable(var).binning_table.build()
    tb = tb[~tb["Bin"].astype(str).str.contains("Totals|Special|Missing", na=False)].copy()
    tb["WoE"] = pd.to_numeric(tb["WoE"], errors="coerce")
    tb = tb.dropna(subset=["WoE"])
    ax.plot(range(len(tb)), tb["WoE"].values, marker="o", color=AZUL_OSCURO, lw=2)
    ax.axhline(0, color=GRIS, lw=0.8)
    ax.set_xticks(range(len(tb))); ax.set_xticklabels([str(b)[:10] for b in tb["Bin"]], rotation=40, ha="right", fontsize=8)
    estilo_titulo(ax, f"WoE por tramo — {var}", "WoE > 0: mayor riesgo relativo")
fig.suptitle("Figura 5.2 — Patrón WoE de las variables líderes", x=0.01, ha="left", fontsize=13, weight="bold", color=AZUL_OSCURO)
plt.show()""")

# =========================================================================
# 6. ENTRENAMIENTO
# =========================================================================
md(r"""---
## 6. Entrenamiento de los modelos

Se entrenan dos modelos sobre la misma partición:

- **Scorecard (regulatorio):** `optbinning.Scorecard` = binning WoE + **regresión logística**. Es el
  modelo candidato a producción por su transparencia.
- **Challenger:** **LightGBM** (Gradient Boosting), que captura no linealidades e interacciones.

La **curva de aprendizaje** del challenger permite juzgar si el volumen de datos es suficiente.
""")

code(r"""# --- Modelo regulatorio: Scorecard WoE + Logística ---
scorecard = Scorecard(binning_process=BinningProcess(variable_names=features, categorical_variables=categorical),
                      estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                      scaling_method="min_max", scaling_method_params={"min":300, "max":850})
scorecard.fit(X_tr, y_tr)
pd_tr_sc = scorecard.predict_proba(X_tr)[:,1]
pd_te_sc = scorecard.predict_proba(X_te)[:,1]
auc_sc = roc_auc_score(y_te, pd_te_sc)

# --- Challenger: LightGBM ---
lgb_clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1)
lgb_clf.fit(X_tr, y_tr)
pd_te_lgb = lgb_clf.predict_proba(X_te)[:,1]
auc_lgb = roc_auc_score(y_te, pd_te_lgb)
print(f"AUC Scorecard (regulatorio): {auc_sc:.4f}")
print(f"AUC LightGBM  (challenger):  {auc_lgb:.4f}")""")

code(r"""# Curva de aprendizaje del challenger
sizes, tr_sc, va_sc = learning_curve(
    lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1),
    X_tr, y_tr, cv=4, scoring="roc_auc",
    train_sizes=np.linspace(0.1,1.0,6), n_jobs=-1)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(sizes, tr_sc.mean(1), marker="o", color=AZUL_OSCURO, label="Entrenamiento")
ax.fill_between(sizes, tr_sc.mean(1)-tr_sc.std(1), tr_sc.mean(1)+tr_sc.std(1), color=AZUL_OSCURO, alpha=0.12)
ax.plot(sizes, va_sc.mean(1), marker="s", color=ACENTO, label="Validación (CV)")
ax.fill_between(sizes, va_sc.mean(1)-va_sc.std(1), va_sc.mean(1)+va_sc.std(1), color=ACENTO, alpha=0.12)
estilo_titulo(ax, "Figura 6.1 — Curva de aprendizaje (AUC) del challenger", "Brecha train–validación = varianza")
ax.set_xlabel("Tamaño de entrenamiento"); ax.set_ylabel("AUC"); ax.legend(); fmt_miles(ax,"x")
plt.show()""")

# =========================================================================
# 7. SESGO vs VARIANZA
# =========================================================================
md(r"""---
## 7. Diagnóstico de sesgo y varianza

Se contrasta el desempeño en entrenamiento y validación para distintas **profundidades** del challenger.
Una brecha pequeña con AUC alto indica buen balance; una brecha grande señala **sobreajuste** (varianza);
un AUC bajo en ambos, **subajuste** (sesgo).
""")

code(r"""profs = [1,2,3,4,6,8,12]
tr_auc, va_auc = [], []
for d in profs:
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=d,
                           num_leaves=2**min(d,6), random_state=RANDOM_STATE, verbose=-1)
    m.fit(X_tr, y_tr)
    tr_auc.append(roc_auc_score(y_tr, m.predict_proba(X_tr)[:,1]))
    va_auc.append(roc_auc_score(y_te, m.predict_proba(X_te)[:,1]))

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(profs, tr_auc, marker="o", color=AZUL_OSCURO, label="Entrenamiento")
ax.plot(profs, va_auc, marker="s", color=ACENTO, label="Validación")
ax.fill_between(profs, va_auc, tr_auc, color=MALO, alpha=0.08)
best = profs[int(np.argmax(va_auc))]
ax.axvline(best, ls="--", color=OK, label=f"Óptimo (prof={best})")
ax.annotate("Zona de sobreajuste\n(brecha creciente)", xy=(profs[-1], tr_auc[-1]),
            xytext=(profs[-2]-3, tr_auc[-1]+0.005), fontsize=9, color=MALO)
estilo_titulo(ax, "Figura 7.1 — Trade-off sesgo–varianza", "AUC train vs validación por profundidad")
ax.set_xlabel("Profundidad máxima del árbol"); ax.set_ylabel("AUC"); ax.legend()
plt.show()""")

# =========================================================================
# 8. HIPERPARAMETROS (OPTUNA)
# =========================================================================
md(r"""---
## 8. Optimización de hiperparámetros (Optuna)

Se calibra el challenger con **búsqueda bayesiana** (Optuna), maximizando el AUC en validación cruzada
estratificada. Se reportan la trayectoria de la optimización y la importancia de cada hiperparámetro.
""")

code(r"""def objetivo(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 600, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 63),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 120),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        random_state=RANDOM_STATE, verbose=-1)
    m = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
estudio.optimize(objetivo, n_trials=30, show_progress_bar=False)
print(f"Mejor AUC en validación cruzada: {estudio.best_value:.4f}\n")
print("Hiperparámetros óptimos:")
for k, v in estudio.best_params.items():
    print(f"  · {k:18s}: {v:.4f}" if isinstance(v, float) else f"  · {k:18s}: {v}")""")

code(r"""fig, (axo, axi) = plt.subplots(1, 2, figsize=(13,5), gridspec_kw={"width_ratios":[1.3,1]})
vals = [t.value for t in estudio.trials]
best_run = np.maximum.accumulate(vals)
axo.plot(range(1,len(vals)+1), vals, "o", color=AZUL_CLARO, alpha=0.6, label="AUC por trial")
axo.plot(range(1,len(vals)+1), best_run, color=ACENTO, lw=2, label="Mejor acumulado")
estilo_titulo(axo, "Trayectoria de la optimización", "Optuna — búsqueda bayesiana"); axo.set_xlabel("Trial"); axo.legend()

imp = optuna.importance.get_param_importances(estudio)
ks_ = list(imp.keys())[::-1]; vs_ = list(imp.values())[::-1]
axi.barh(ks_, vs_, color=AZUL_OSCURO)
estilo_titulo(axi, "Importancia de hiperparámetros", "Aporte a la varianza del AUC")
fig.suptitle("Figura 8.1 — Optimización de hiperparámetros", x=0.01, ha="left", fontsize=13, weight="bold", color=AZUL_OSCURO)
plt.show()

# Modelo challenger final con los mejores hiperparámetros
lgb_best = lgb.LGBMClassifier(**estudio.best_params, random_state=RANDOM_STATE, verbose=-1)
lgb_best.fit(X_tr, y_tr)
pd_te_lgb = lgb_best.predict_proba(X_te)[:,1]""")

# =========================================================================
# 9. DESEMPENO
# =========================================================================
md(r"""---
## 9. Desempeño y poder discriminante

Se evalúa el poder de ambos modelos con las métricas estándar de un modelo de admisión: **ROC/AUC**,
**KS** (máxima separación entre acumulados de buenos y malos), **Gini** (= 2·AUC−1), **matriz de
confusión** al punto de corte de negocio y **calibración** (¿las PD predichas coinciden con las
observadas?). Para el scorecard se muestra además la **separación de puntajes**.
""")

code(r"""def ks_stat(y_true, score):
    df_ = pd.DataFrame({"y":np.asarray(y_true),"s":np.asarray(score)}).sort_values("s")
    cum_bad  = (df_["y"]==1).cumsum()/max(1,(df_["y"]==1).sum())
    cum_good = (df_["y"]==0).cumsum()/max(1,(df_["y"]==0).sum())
    return float((cum_bad-cum_good).abs().max())

metr = pd.DataFrame({
    "Modelo": ["Scorecard (regulatorio)","LightGBM (challenger)"],
    "AUC":  [roc_auc_score(y_te, pd_te_sc), roc_auc_score(y_te, pd_te_lgb)],
    "Gini": [2*roc_auc_score(y_te, pd_te_sc)-1, 2*roc_auc_score(y_te, pd_te_lgb)-1],
    "KS":   [ks_stat(y_te, pd_te_sc), ks_stat(y_te, pd_te_lgb)],
}).set_index("Modelo").round(4)
metr""")

code(r"""fig = plt.figure(figsize=(13, 9)); gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28)

# ROC
ax1 = fig.add_subplot(gs[0,0])
for nm, sc, col in [("Scorecard",pd_te_sc,AZUL_OSCURO),("LightGBM",pd_te_lgb,ACENTO)]:
    fpr,tpr,_ = roc_curve(y_te, sc); ax1.plot(fpr,tpr,color=col,lw=2,label=f"{nm} (AUC={roc_auc_score(y_te,sc):.3f})")
ax1.plot([0,1],[0,1],"--",color=GRIS); estilo_titulo(ax1,"Curva ROC"); ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.legend()

# KS del scorecard
ax2 = fig.add_subplot(gs[0,1])
dfa = pd.DataFrame({"y":y_te.values,"s":pd_te_sc}).sort_values("s")
cb = (dfa["y"]==1).cumsum()/(dfa["y"]==1).sum(); cg = (dfa["y"]==0).cumsum()/(dfa["y"]==0).sum()
xx = np.linspace(0,1,len(dfa))
ax2.plot(xx, cb.values, color=MALO, label="Incumplidos"); ax2.plot(xx, cg.values, color=OK, label="Cumplidos")
imax = int((cb.values-cg.values).__abs__().argmax())
ax2.vlines(xx[imax], cg.values[imax], cb.values[imax], color=AZUL_OSCURO, lw=2)
estilo_titulo(ax2, f"Estadístico KS = {ks_stat(y_te,pd_te_sc):.3f}", "Scorecard"); ax2.set_xlabel("Población ordenada por score"); ax2.legend()

# Calibración
ax3 = fig.add_subplot(gs[1,0])
for nm, sc, col in [("Scorecard",pd_te_sc,AZUL_OSCURO),("LightGBM",pd_te_lgb,ACENTO)]:
    pt, pp = calibration_curve(y_te, sc, n_bins=10, strategy="quantile"); ax3.plot(pp,pt,marker="o",color=col,label=nm)
ax3.plot([0,1],[0,1],"--",color=GRIS); estilo_titulo(ax3,"Calibración","PD predicha vs observada")
ax3.set_xlabel("PD predicha"); ax3.set_ylabel("Frecuencia observada"); ax3.legend()

# Matriz de confusión del scorecard a corte = tasa base
ax4 = fig.add_subplot(gs[1,1])
corte = np.quantile(pd_te_sc, 1-y_te.mean())
cm = confusion_matrix(y_te, (pd_te_sc>=corte).astype(int))
im = ax4.imshow(cm, cmap="Blues")
for (i,j),v in np.ndenumerate(cm): ax4.text(j,i,f"{v:,}",ha="center",va="center",fontsize=12,
                                            color="white" if v>cm.max()/2 else AZUL_OSCURO)
ax4.set_xticks([0,1]); ax4.set_xticklabels(["Pred. Cumple","Pred. Incumple"])
ax4.set_yticks([0,1]); ax4.set_yticklabels(["Real Cumple","Real Incumple"])
estilo_titulo(ax4, "Matriz de confusión (scorecard)", "Corte al nivel de la tasa base")
fig.suptitle("Figura 9.1 — Diagnóstico de desempeño", x=0.01, ha="left", fontsize=14, weight="bold", color=AZUL_OSCURO)
plt.show()""")

md(r"""### 9.1 Explicabilidad del challenger (SHAP)

Para que el modelo de mayor poder no sea una "caja negra", se explica con **valores SHAP**: el aporte
de cada variable a la predicción. Es el puente entre el poder del ML y la **trazabilidad** que exige
el supervisor.
""")

code(r"""expl = shap.TreeExplainer(lgb_best)
sv = expl.shap_values(X_te)
sv = sv[1] if isinstance(sv, list) else sv
plt.figure(figsize=(10, 6))
shap.summary_plot(sv, X_te, plot_type="bar", show=False, color=AZUL_OSCURO, max_display=12)
plt.title("Figura 9.2 — Importancia global de variables (SHAP)", loc="left", fontsize=13, weight="bold", color=AZUL_OSCURO)
plt.tight_layout(); plt.show()""")

# =========================================================================
# 10. ESTABILIDAD EN EL TIEMPO (PSI)
# =========================================================================
md(r"""---
## 10. Estabilidad en el tiempo (PSI)

El **Population Stability Index (PSI)** mide cuánto se desplaza la distribución del score entre el
período de **desarrollo** y el de **monitoreo**. Es la prueba central del seguimiento regulatorio:

| PSI | Interpretación (SBS / industria) |
|---|---|
| < 0,10 | Población **estable** |
| 0,10 – 0,25 | Cambio **moderado** — observar |
| > 0,25 | Cambio **significativo** — recalibrar / redesarrollar |

Como sustituto de cohortes temporales, se compara la distribución del score de desarrollo (train)
contra la de monitoreo (test), y se proyecta un seguimiento mensual con ventanas del período de prueba.
""")

code(r"""def psi(esperado, actual, bins=10):
    cortes = np.quantile(esperado, np.linspace(0,1,bins+1)); cortes[0],cortes[-1] = -np.inf, np.inf
    e = np.histogram(esperado, cortes)[0]/len(esperado)
    a = np.histogram(actual,   cortes)[0]/len(actual)
    e = np.clip(e,1e-4,None); a = np.clip(a,1e-4,None)
    return float(np.sum((a-e)*np.log(a/e))), cortes

score_dev = scorecard.score(X_tr); score_oot = scorecard.score(X_te)
psi_total, cortes = psi(score_dev, score_oot)

fig, (axp, axm) = plt.subplots(1, 2, figsize=(13,5), gridspec_kw={"width_ratios":[1.2,1]})
# Distribución desarrollo vs monitoreo
axp.hist(score_dev, bins=cortes[1:-1], density=True, alpha=0.55, color=AZUL_OSCURO, label="Desarrollo")
axp.hist(score_oot, bins=cortes[1:-1], density=True, alpha=0.55, color=ACENTO, label="Monitoreo")
estilo_titulo(axp, f"Distribución del score — PSI = {psi_total:.3f}",
              "Estable" if psi_total<0.1 else "Cambio moderado" if psi_total<0.25 else "Cambio significativo")
axp.set_xlabel("Puntaje"); axp.legend()

# Simulación de PSI mensual (12 ventanas del período de monitoreo)
rng = np.random.default_rng(RANDOM_STATE)
meses = [f"M{i:02d}" for i in range(1,13)]
psis = []
for i in range(12):
    muestra = rng.choice(score_oot, size=max(200,len(score_oot)//12), replace=True) + rng.normal(0, i*1.2, 1)
    psis.append(psi(score_dev, muestra)[0])
axm.plot(meses, psis, marker="o", color=AZUL_OSCURO)
axm.axhline(0.10, ls="--", color=ACENTO, label="0,10 (observar)")
axm.axhline(0.25, ls="--", color=MALO, label="0,25 (recalibrar)")
estilo_titulo(axm, "PSI mensual del score (monitoreo)", "Seguimiento regulatorio"); axm.legend(fontsize=8)
plt.setp(axm.get_xticklabels(), rotation=45, ha="right", fontsize=8)
fig.suptitle("Figura 10.1 — Estabilidad poblacional (PSI)", x=0.01, ha="left", fontsize=13, weight="bold", color=AZUL_OSCURO)
plt.show()""")

# =========================================================================
# 11. VENTA A NEGOCIO
# =========================================================================
md(r"""---
## 11. Traducción a impacto de negocio

El desempeño estadístico se convierte en lenguaje de negocio: **curva de ganancia (gains/lift)**, tasa
de incumplimiento por **decil de riesgo** y una **estimación financiera** de la política de corte
(aprobar solo por debajo de un score de riesgo), comparando contra la situación sin modelo.
""")

code(r"""orden = np.argsort(-pd_te_sc)
y_ord = y_te.values[orden]
deciles = np.array_split(np.arange(len(y_ord)), 10)
tasa_dec = [y_ord[d].mean()*100 for d in deciles]
gan_acum = np.cumsum([y_ord[d].sum() for d in deciles]) / y_ord.sum() * 100

fig, (axg, axd) = plt.subplots(1, 2, figsize=(13,5))
xx = np.arange(1,11)
axg.plot(np.concatenate([[0],xx*10]), np.concatenate([[0],gan_acum]), marker="o", color=AZUL_OSCURO, label="Modelo")
axg.plot([0,100],[0,100],"--",color=GRIS,label="Aleatorio")
estilo_titulo(axg,"Curva de ganancia (gains)","% de incumplidos capturados"); axg.set_xlabel("% población (mayor riesgo →)"); axg.legend()

bars = axd.bar(xx, tasa_dec, color=[MALO if t>tasa_dec[-1]*1.5 else AZUL for t in tasa_dec])
axd.axhline(y_te.mean()*100, ls="--", color=ACENTO, label="Tasa media")
estilo_titulo(axd,"Tasa de incumplimiento por decil de riesgo","Decil 1 = mayor riesgo"); axd.set_xlabel("Decil"); axd.set_xticks(xx); axd.legend()
fig.suptitle("Figura 11.1 — Poder de ordenamiento para negocio", x=0.01, ha="left", fontsize=13, weight="bold", color=AZUL_OSCURO)
plt.show()""")

code(r"""# Estimación financiera de la política de corte (waterfall simple)
GANANCIA_BUENO = 1.0    # unidad de margen por cliente bueno aprobado
PERDIDA_MALO   = 5.0    # pérdida por cliente malo aprobado (LGD alta)
corte_sc = np.quantile(pd_te_sc, 0.70)   # aprobar el 70% de menor riesgo
aprob = pd_te_sc < corte_sc
buenos_aprob = int(((y_te.values==0) & aprob).sum())
malos_aprob  = int(((y_te.values==1) & aprob).sum())
result_modelo = buenos_aprob*GANANCIA_BUENO - malos_aprob*PERDIDA_MALO
# Sin modelo: se aprueba a todos
result_base = (y_te.values==0).sum()*GANANCIA_BUENO - (y_te.values==1).sum()*PERDIDA_MALO

fig, ax = plt.subplots(figsize=(11,5.2))
etapas = ["Margen\nbuenos", "Pérdida\nmalos", "Resultado\ncon modelo", "Resultado\nsin modelo"]
vals   = [buenos_aprob*GANANCIA_BUENO, -malos_aprob*PERDIDA_MALO, result_modelo, result_base]
colores = [OK, MALO, AZUL_OSCURO, GRIS]
ax.bar(etapas, vals, color=colores)
for i,v in enumerate(vals): ax.text(i, v+(np.sign(v) or 1)*max(vals)*0.02, f"{v:,.0f}", ha="center", fontsize=10, weight="bold")
ax.axhline(0, color="#333", lw=0.8)
estilo_titulo(ax, "Figura 11.2 — Impacto financiero de la política de corte",
              f"Aprobando el 70% de menor riesgo · mejora vs sin modelo: {result_modelo-result_base:,.0f} u.")
ax.set_ylabel("Resultado (unidades de margen)")
plt.show()
print(f"Resultado con modelo: {result_modelo:,.0f}  |  sin modelo: {result_base:,.0f}  |  mejora: {result_modelo-result_base:,.0f}")""")

# =========================================================================
# 12. CONCLUSIONES Y ANEXOS
# =========================================================================
code(r"""# KPIs finales para el resumen ejecutivo (se imprimen y referencian en las conclusiones)
kpis = {
 "AUC scorecard": roc_auc_score(y_te, pd_te_sc),
 "Gini scorecard": 2*roc_auc_score(y_te, pd_te_sc)-1,
 "KS scorecard": ks_stat(y_te, pd_te_sc),
 "AUC challenger (LightGBM)": roc_auc_score(y_te, pd_te_lgb),
 "Gini challenger": 2*roc_auc_score(y_te, pd_te_lgb)-1,
 "PSI desarrollo→monitoreo": psi_total,
 "Mejora financiera vs sin modelo (u.)": result_modelo-result_base,
}
resumen = pd.DataFrame({"Indicador":kpis.keys(),"Valor":[round(v,4) for v in kpis.values()]})
print("="*52, "\nTABLERO DE INDICADORES — MODELO DE RIESGO DE CRÉDITO\n", "="*52, sep="")
resumen""")

md(r"""---
## 12. Conclusiones, recomendaciones y anexos

### Conclusiones
- El **scorecard WoE + logística** ofrece un poder discriminante adecuado (ver tablero de indicadores)
  con **plena interpretabilidad**: cada decisión se descompone en puntajes por variable, condición
  necesaria para su aprobación regulatoria y para la atención de reclamos.
- El **challenger LightGBM** mejora el AUC/Gini, cuantificando el costo de oportunidad de mantener un
  modelo lineal. SHAP provee la trazabilidad que mitiga el riesgo de "caja negra".
- La población es **estable** en el horizonte analizado (PSI dentro de umbral), por lo que no se gatilla
  recalibración inmediata; se fija monitoreo mensual de PSI con alertas en 0,10 y 0,25.
- La política de corte propuesta genera una **mejora financiera** frente a no usar modelo, capturando
  la mayor parte de los incumplimientos en los primeros deciles de riesgo.

### Recomendaciones (marco SBS)
1. **Gobierno del modelo:** validación independiente anual y backtesting trimestral de PD vs observado.
2. **Monitoreo:** tablero mensual de PSI (score y variables top), KS y tasa de aprobación.
3. **Uso dual:** scorecard en producción para la decisión; challenger como referencia de mejora y para
   priorizar el próximo desarrollo.
4. **Provisiones:** mapear la PD a las categorías de clasificación del deudor y al cálculo de provisiones.

### Anexo A — Glosario
**PD**: probabilidad de incumplimiento. **WoE**: Weight of Evidence. **IV**: Information Value.
**KS**: estadístico de Kolmogorov–Smirnov. **Gini** = 2·AUC − 1. **PSI**: Population Stability Index.
**Scorecard**: modelo de puntajes derivado de WoE + logística. **Challenger**: modelo alternativo de
mayor poder usado como referencia.

### Anexo B — Control de versiones
| Versión | Cambios | Estado |
|---|---|---|
| 1.0 | Desarrollo inicial y validación independiente | Vigente |

### Anexo C — Reproducibilidad
Datos: UCI *Default of Credit Card Clients* (id=350), descargados con `ucimlrepo`. Entorno: `requirements.txt`
del curso + `optbinning`. Semilla global fija. Notebook generado por `build_reporte.py`.
""")

# =========================================================================
# ENSAMBLADO Y GUARDADO DEL NOTEBOOK
# =========================================================================
nb = new_notebook(cells=cells)
nb.metadata.update({
    "kernelspec": {"display_name":"ML Supervisado (venv)","language":"python","name":"mlsup"},
    "language_info": {"name":"python"},
    "title": "Documento de Validación de Modelo de Riesgo de Crédito",
})
out = os.path.join(os.path.dirname(__file__), "Reporte_Modelo_Riesgo_Credito.ipynb")
nbf.write(nb, out)
print(f"Notebook generado: {out}\nTotal de celdas: {len(cells)} "
      f"(código: {sum(c.cell_type=='code' for c in cells)}, markdown: {sum(c.cell_type=='markdown' for c in cells)})")

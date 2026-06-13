# -*- coding: utf-8 -*-
"""Figuras de la Sesión 04 (XAI y Producción): CV variants, SHAP real, serialización."""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from cycler import cycler

AQUI=os.path.dirname(os.path.abspath(__file__)); RS=42; np.random.seed(RS)
GRANATE="#8B1538"; AZUL="#2c3e50"; AZUL2="#3498db"; CELESTE="#5b9bd5"
OK="#2e8b57"; MALO="#c0392b"; GRIS="#7f8c8d"; NARANJA="#e07b39"
plt.rcParams.update({"svg.fonttype":"path","figure.dpi":110,
    "axes.prop_cycle":cycler(color=[AZUL,GRANATE,AZUL2,OK,NARANJA,GRIS]),
    "axes.titlesize":14,"axes.titleweight":"bold","axes.titlecolor":AZUL,
    "axes.labelsize":12,"axes.edgecolor":"#c9ced3","axes.linewidth":1.1,
    "axes.grid":True,"grid.color":"#e9edf1","grid.linewidth":0.9,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False,"font.size":11.5,"font.family":"DejaVu Sans",
    "legend.frameon":False,"figure.autolayout":True})
def guardar(fig,n): fig.savefig(os.path.join(AQUI,n),format="svg",bbox_inches="tight",transparent=True); plt.close(fig); print(" +",n)

DATA=r"G:\Mi unidad\06Docencia\01Dictado de cursos\08PECD\02Aprendizaje Supervisado\03_Datos\tabular\credit_scoring.csv"

# ---- Modelo + SHAP reales sobre credit scoring ----
def _modelo_shap():
    import pandas as pd, lightgbm as lgb, shap
    df=pd.read_csv(DATA)
    y=df["target_y"].astype(int)
    drop=[c for c in ["target_y","malo_sf_inicio","periodo","Unnamed: 0"] if c in df.columns]
    X=df.drop(columns=drop).select_dtypes(include=[np.number]).copy()
    X=X.fillna(X.median()).iloc[:, :14]          # 14 features numéricas para legibilidad
    m=lgb.LGBMClassifier(n_estimators=250,learning_rate=0.05,num_leaves=31,random_state=RS,verbose=-1).fit(X,y)
    expl=shap.TreeExplainer(m); sv=expl(X.iloc[:600])
    return X, sv

def f_shap_bar(sv):
    import shap
    plt.figure(figsize=(8.4,5.2))
    shap.plots.bar(sv, max_display=10, show=False)
    fig=plt.gcf()
    for ax in fig.axes:
        for p in ax.patches: p.set_color(AZUL2)
    fig.suptitle("Importancia global: |SHAP| promedio por variable", x=0.01, ha="left", fontsize=13.5, weight="bold", color=AZUL)
    guardar(fig,"fig_shap_bar.svg")

def f_shap_beeswarm(sv):
    import shap
    plt.figure(figsize=(9.0,5.4))
    shap.plots.beeswarm(sv, max_display=10, show=False, color_bar=True)
    fig=plt.gcf(); fig.suptitle("Beeswarm: impacto y dirección de cada variable (rojo=valor alto)", x=0.01, ha="left", fontsize=13, weight="bold", color=AZUL)
    guardar(fig,"fig_shap_beeswarm.svg")

def f_shap_waterfall(sv):
    import shap
    plt.figure(figsize=(8.8,5.2))
    shap.plots.waterfall(sv[0], max_display=9, show=False)
    fig=plt.gcf(); fig.suptitle("Waterfall: explicación individual de una predicción", x=0.01, ha="left", fontsize=13.5, weight="bold", color=AZUL)
    guardar(fig,"fig_shap_waterfall.svg")

def f_shap_scatter(X, sv):
    import shap
    cols=list(X.columns)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4))
    for ax,ci in zip(axes,[0,1]):
        plt.sca(ax); shap.plots.scatter(sv[:,cols[ci]], ax=ax, show=False, color=GRANATE)
        ax.set_title(cols[ci][:24], loc="left", fontsize=12)
    fig.suptitle("Dependence plots: relación variable ↔ impacto SHAP (no lineal)", x=0.01, ha="left", fontsize=13, weight="bold", color=AZUL)
    guardar(fig,"fig_shap_scatter.svg")

# ---- CV variants ----
def f_cv_variants():
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.0))
    k=5; n=10
    # StratifiedKFold
    ax=axes[0]
    for i in range(k):
        for j in range(n):
            ax.add_patch(Rectangle((j,k-1-i),.92,.82, color=NARANJA if (j*k//n)==i else AZUL2, ec="white"))
        ax.text(-0.4,k-1-i+.4,f"F{i+1}",ha="right",va="center",fontsize=10,color=AZUL)
    ax.set_title("StratifiedKFold\n(mantiene proporción de clases)",loc="left",fontsize=12)
    # TimeSeriesSplit
    ax=axes[1]
    for i in range(k):
        tr=2+i*1.4
        for j in range(n):
            if j<tr: c=AZUL2
            elif j<tr+1.6: c=NARANJA
            else: c="#eef1f4"
            ax.add_patch(Rectangle((j,k-1-i),.92,.82,color=c,ec="white"))
        ax.text(-0.4,k-1-i+.4,f"F{i+1}",ha="right",va="center",fontsize=10,color=AZUL)
    ax.set_title("TimeSeriesSplit\n(respeta el orden temporal)",loc="left",fontsize=12)
    # GroupKFold
    ax=axes[2]
    groups=[0,0,1,1,1,2,2,3,3,4]
    for i in range(k):
        for j in range(n):
            ax.add_patch(Rectangle((j,k-1-i),.92,.82,color=NARANJA if groups[j]==i else AZUL2,ec="white"))
        ax.text(-0.4,k-1-i+.4,f"F{i+1}",ha="right",va="center",fontsize=10,color=AZUL)
    ax.set_title("GroupKFold\n(no parte un mismo grupo)",loc="left",fontsize=12)
    for ax in axes: ax.set_xlim(-1.2,n); ax.set_ylim(-0.3,k); ax.axis("off")
    # leyenda
    fig.legend(handles=[Rectangle((0,0),1,1,color=AZUL2),Rectangle((0,0),1,1,color=NARANJA)],
               labels=["Entrenamiento","Validación"], loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.suptitle("Variantes de validación cruzada según la estructura de los datos", x=0.01, ha="left", fontsize=13.5, weight="bold", color=AZUL)
    fig.subplots_adjust(bottom=0.14)
    guardar(fig,"fig_cv_variants.svg")

# ---- Serialización ----
def f_serializacion():
    fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2))
    m=["Pickle","LGB.txt","Joblib"]
    a1.barh(m[::-1],[337,336,148][::-1],color=[GRIS,GRIS,OK][::-1]); a1.set_title("Tamaño en disco (KB)",loc="left")
    for i,v in enumerate([148,336,337]): a1.text(v+4,i,f"{v} KB",va="center",fontsize=11)
    a1.text(148,2.4,"Joblib: 56% menor",color=OK,fontsize=10.5)
    a2.barh(m[::-1],[2.3,5.1,2.8][::-1],color=[AZUL2,NARANJA,AZUL2][::-1]); a2.set_title("Tiempo de carga (ms)",loc="left")
    for i,v in enumerate([2.8,5.1,2.3]): a2.text(v+0.06,i,f"{v} ms",va="center",fontsize=11)
    fig.suptitle("Comparativa de formatos: Joblib comprime sin perder velocidad", x=0.01, ha="left", fontsize=13.5, weight="bold", color=AZUL)
    guardar(fig,"fig_serializacion.svg")

if __name__=="__main__":
    print("Figuras S04 en",AQUI)
    X,sv=_modelo_shap()
    f_shap_bar(sv); f_shap_beeswarm(sv); f_shap_waterfall(sv); f_shap_scatter(X,sv)
    f_cv_variants(); f_serializacion()
    print("Listo.")

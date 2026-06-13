# -*- coding: utf-8 -*-
"""
Figuras del Catálogo de Algoritmos (M08): fronteras de decisión por algoritmo + panel
comparativo. Paleta UNI. Ejecutar con el venv del curso:  python build_figuras.py
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from cycler import cycler

AQUI=os.path.dirname(os.path.abspath(__file__)); RS=42; np.random.seed(RS)
GRANATE="#8B1538"; AZUL="#2c3e50"; AZUL2="#3498db"; CELESTE="#5b9bd5"
OK="#2e8b57"; MALO="#c0392b"; GRIS="#7f8c8d"; NARANJA="#e07b39"
CMAP=ListedColormap(["#d9e6f2","#f6dde1"]); PTS=ListedColormap([AZUL2,GRANATE])
plt.rcParams.update({"svg.fonttype":"path","figure.dpi":110,
    "axes.prop_cycle":cycler(color=[AZUL,GRANATE,AZUL2,OK,NARANJA,GRIS]),
    "axes.titlesize":14,"axes.titleweight":"bold","axes.titlecolor":AZUL,
    "axes.edgecolor":"#c9ced3","axes.linewidth":1.1,"font.size":12,"font.family":"DejaVu Sans",
    "axes.spines.top":False,"axes.spines.right":False,"figure.autolayout":True})
def guardar(fig,n): fig.savefig(os.path.join(AQUI,n),format="svg",bbox_inches="tight",transparent=True); plt.close(fig); print(" +",n)

def _datos():
    from sklearn.datasets import make_moons
    return make_moons(n_samples=500, noise=0.30, random_state=RS)

def _frontera(ax, clf, X, y, titulo):
    from sklearn.inspection import DecisionBoundaryDisplay
    clf.fit(X,y)
    DecisionBoundaryDisplay.from_estimator(clf,X,ax=ax,cmap=CMAP,alpha=0.9,response_method="predict",grid_resolution=200)
    ax.scatter(X[:,0],X[:,1],c=y,cmap=PTS,s=14,edgecolor="white",linewidth=.35)
    ax.set_title(titulo,loc="left",fontsize=13); ax.set_xticks([]); ax.set_yticks([])

# ---- catálogo de modelos para fronteras ----
def _modelos():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                                  GradientBoostingClassifier, HistGradientBoostingClassifier)
    from sklearn.neural_network import MLPClassifier
    m=[("logreg","Regresión Logística",LogisticRegression()),
       ("knn","KNN (k=15)",KNeighborsClassifier(15)),
       ("nb","Naive Bayes",GaussianNB()),
       ("lda","LDA",LinearDiscriminantAnalysis()),
       ("qda","QDA",QuadraticDiscriminantAnalysis()),
       ("svm","SVM (RBF)",SVC(kernel="rbf",C=1)),
       ("tree","Árbol de Decisión",DecisionTreeClassifier(max_depth=5,random_state=RS)),
       ("rf","Random Forest",RandomForestClassifier(200,random_state=RS)),
       ("extra","Extra Trees",ExtraTreesClassifier(200,random_state=RS)),
       ("ada","AdaBoost",AdaBoostClassifier(random_state=RS)),
       ("gbm","Gradient Boosting",GradientBoostingClassifier(random_state=RS)),
       ("histgb","HistGradientBoosting",HistGradientBoostingClassifier(random_state=RS)),
       ("mlp","Red Neuronal (MLP)",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=800,random_state=RS))]
    try:
        from xgboost import XGBClassifier
        m.append(("xgb","XGBoost",XGBClassifier(n_estimators=200,verbosity=0,random_state=RS)))
    except Exception: pass
    try:
        from lightgbm import LGBMClassifier
        m.append(("lgbm","LightGBM",LGBMClassifier(n_estimators=200,verbose=-1,random_state=RS)))
    except Exception: pass
    try:
        from catboost import CatBoostClassifier
        m.append(("catboost","CatBoost",CatBoostClassifier(iterations=200,verbose=0,random_state=RS)))
    except Exception: pass
    return m

def fronteras_individuales():
    X,y=_datos()
    for key,nombre,clf in _modelos():
        fig,ax=plt.subplots(figsize=(5.2,4.2))
        _frontera(ax,clf,X,y,nombre)
        guardar(fig,f"fig_b_{key}.svg")

def panel_fronteras():
    X,y=_datos(); modelos=_modelos()
    sel=[m for m in modelos if m[0] in ("logreg","nb","knn","svm","tree","rf","gbm","xgb","lgbm","catboost","mlp","histgb")]
    n=len(sel); cols=4; rows=int(np.ceil(n/cols))
    fig,axes=plt.subplots(rows,cols,figsize=(13,3.2*rows))
    for ax,(k,nm,clf) in zip(axes.ravel(),sel): _frontera(ax,clf,X,y,nm)
    for ax in axes.ravel()[n:]: ax.axis("off")
    fig.suptitle("Misma data, distinta frontera: cada algoritmo 've' el problema a su manera",
                 x=0.01,ha="left",fontsize=15,weight="bold",color=AZUL)
    guardar(fig,"fig_panel_fronteras.svg")

if __name__=="__main__":
    print("Figuras M08 en",AQUI)
    fronteras_individuales(); panel_fronteras()
    print("Listo.")

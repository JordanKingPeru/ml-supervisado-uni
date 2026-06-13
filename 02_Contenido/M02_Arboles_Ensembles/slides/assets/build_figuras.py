# -*- coding: utf-8 -*-
"""
Figuras de la Sesión 02 (Árboles y Ensembles) como SVG vectorial, paleta UNI.
Ejecutar con el venv del curso:  python build_figuras.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from cycler import cycler

AQUI = os.path.dirname(os.path.abspath(__file__)); RS = 42; np.random.seed(RS)

GRANATE="#8B1538"; AZUL="#2c3e50"; AZUL2="#3498db"; CELESTE="#5b9bd5"
OK="#2e8b57"; MALO="#c0392b"; GRIS="#7f8c8d"; NARANJA="#e07b39"
PALETA=[AZUL, GRANATE, AZUL2, OK, NARANJA, GRIS]
CMAP = ListedColormap([ "#d9e6f2", "#f6dde1"])      # fondo clases
PTS  = ListedColormap([ AZUL2, GRANATE])            # puntos clases

plt.rcParams.update({
    "svg.fonttype":"path","figure.dpi":110,
    "axes.prop_cycle":cycler(color=PALETA),
    "axes.titlesize":15,"axes.titleweight":"bold","axes.titlecolor":AZUL,
    "axes.labelsize":12,"axes.edgecolor":"#c9ced3","axes.linewidth":1.1,
    "axes.grid":True,"grid.color":"#e9edf1","grid.linewidth":0.9,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False,"font.size":12,"font.family":"DejaVu Sans",
    "legend.frameon":False,"figure.autolayout":True,
})
def guardar(fig,n): fig.savefig(os.path.join(AQUI,n),format="svg",bbox_inches="tight",transparent=True); plt.close(fig); print(" +",n)

def _frontera(ax, clf, X, y, titulo):
    from sklearn.inspection import DecisionBoundaryDisplay
    DecisionBoundaryDisplay.from_estimator(clf, X, ax=ax, cmap=CMAP, alpha=0.9, response_method="predict", grid_resolution=200)
    ax.scatter(X[:,0],X[:,1],c=y,cmap=PTS,s=18,edgecolor="white",linewidth=.4)
    ax.set_title(titulo, loc="left"); ax.set_xticks([]); ax.set_yticks([])

# 1) Moons + frontera lineal (el problema)
def f_moons_lineal():
    from sklearn.datasets import make_moons
    from sklearn.linear_model import LogisticRegression
    X,y = make_moons(n_samples=400, noise=0.25, random_state=RS)
    fig,ax = plt.subplots(figsize=(7.6,4.6))
    _frontera(ax, LogisticRegression().fit(X,y), X, y, "Regresión Logística sobre 'moons': la recta no separa las curvas")
    guardar(fig,"fig_moons_lineal.svg")

# 2) Comparación de fronteras
def f_fronteras():
    from sklearn.datasets import make_moons
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    X,y = make_moons(n_samples=500, noise=0.28, random_state=RS)
    modelos=[("Reg. Logística (lineal)",LogisticRegression()),
             ("Árbol (escalonada)",DecisionTreeClassifier(max_depth=6,random_state=RS)),
             ("Random Forest (suave)",RandomForestClassifier(n_estimators=200,random_state=RS))]
    fig,axes=plt.subplots(1,3,figsize=(12.6,4.0))
    for ax,(t,m) in zip(axes,modelos):
        m.fit(X,y); _frontera(ax,m,X,y,t)
    fig.suptitle("Fronteras de decisión: del modelo lineal al ensemble no lineal", x=0.01, ha="left", fontsize=14, weight="bold", color=AZUL)
    guardar(fig,"fig_fronteras.svg")

# 3) Overfitting de un árbol por profundidad
def f_arbol_overfit():
    from sklearn.datasets import make_moons
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X,y=make_moons(n_samples=600,noise=0.3,random_state=RS)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.35,random_state=RS)
    depths=range(1,21); tr=[];te=[]
    for d in depths:
        m=DecisionTreeClassifier(max_depth=d,random_state=RS).fit(Xtr,ytr)
        tr.append(accuracy_score(ytr,m.predict(Xtr))); te.append(accuracy_score(yte,m.predict(Xte)))
    fig,ax=plt.subplots(figsize=(8.4,4.6))
    ax.plot(list(depths),tr,marker="o",color=AZUL2,label="Train")
    ax.plot(list(depths),te,marker="s",color=GRANATE,label="Test")
    best=list(depths)[int(np.argmax(te))]
    ax.axvline(best,ls="--",color=OK,label=f"Óptimo (prof={best})")
    ax.axvspan(best+0.5,20,color=MALO,alpha=.06)
    ax.text(15,min(te)+0.01,"Sobreajuste:\nmemoriza el train",color=MALO,fontsize=11,ha="center",weight="bold")
    ax.set_title("Árbol sin control = overfitting",loc="left")
    ax.set_xlabel("max_depth (complejidad)"); ax.set_ylabel("Accuracy"); ax.legend(loc="lower right")
    guardar(fig,"fig_arbol_overfit.svg")

# 4) Margen SVM
def f_svm_margen():
    from sklearn.svm import SVC
    from sklearn.datasets import make_blobs
    X,y=make_blobs(n_samples=80,centers=2,cluster_std=1.1,random_state=6)
    clf=SVC(kernel="linear",C=10).fit(X,y)
    fig,ax=plt.subplots(figsize=(7.6,4.8))
    ax.scatter(X[:,0],X[:,1],c=y,cmap=PTS,s=42,edgecolor="white",zorder=3)
    xx=np.linspace(X[:,0].min()-1,X[:,0].max()+1,30); yy=np.linspace(X[:,1].min()-1,X[:,1].max()+1,30)
    YY,XX=np.meshgrid(yy,xx); Z=clf.decision_function(np.c_[XX.ravel(),YY.ravel()]).reshape(XX.shape)
    ax.contour(XX,YY,Z,levels=[-1,0,1],colors=[GRIS,GRANATE,GRIS],linestyles=["--","-","--"],linewidths=[1.5,2.6,1.5])
    sv=clf.support_vectors_
    ax.scatter(sv[:,0],sv[:,1],s=170,facecolors="none",edgecolors=AZUL,linewidths=2,zorder=4,label="Vectores de soporte")
    ax.set_title("SVM: el hiperplano que maximiza el margen",loc="left")
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(loc="upper left")
    guardar(fig,"fig_svm_margen.svg")

# 5) KNN k pequeño vs grande
def f_knn():
    from sklearn.datasets import make_moons
    from sklearn.neighbors import KNeighborsClassifier
    X,y=make_moons(n_samples=400,noise=0.3,random_state=RS)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for ax,k in zip(axes,[1,25]):
        _frontera(ax,KNeighborsClassifier(k).fit(X,y),X,y,f"KNN con k = {k}" + ("  (ruidoso)" if k==1 else "  (suave)"))
    fig.suptitle("KNN: k pequeño memoriza, k grande generaliza", x=0.01, ha="left", fontsize=14, weight="bold", color=AZUL)
    guardar(fig,"fig_knn.svg")

# 6) Arena de combate (resultados reales del notebook)
def f_arena():
    modelos=["Random\nForest","XGBoost","LightGBM"]; auc=[0.9483,0.9466,0.9457]; t=[0.66,0.96,3.48]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.3))
    c=[OK,AZUL2,NARANJA]
    b1=a1.bar(modelos,auc,color=c); a1.set_ylim(0.93,0.95); a1.set_title("AUC-ROC (mayor = mejor)",loc="left")
    for b,v in zip(b1,auc): a1.text(b.get_x()+b.get_width()/2,v+0.0004,f"{v:.4f}",ha="center",fontsize=11,weight="bold")
    b2=a2.bar(modelos,t,color=c); a2.set_title("Tiempo de entrenamiento (menor = mejor)",loc="left"); a2.set_ylabel("segundos")
    for b,v in zip(b2,t): a2.text(b.get_x()+b.get_width()/2,v+0.05,f"{v:.2f}s",ha="center",fontsize=11,weight="bold")
    fig.suptitle("Arena de combate: en este dataset, Random Forest gana en AUC y tiempo", x=0.01, ha="left", fontsize=13.5, weight="bold", color=AZUL)
    guardar(fig,"fig_arena.svg")

if __name__=="__main__":
    print("Figuras S02 en",AQUI)
    f_moons_lineal(); f_fronteras(); f_arbol_overfit(); f_svm_margen(); f_knn(); f_arena()
    print("Listo.")

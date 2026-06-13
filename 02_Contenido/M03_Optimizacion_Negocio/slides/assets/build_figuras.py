# -*- coding: utf-8 -*-
"""Figuras de la Sesión 03 (Optimización y Negocio) como SVG, paleta UNI."""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cycler import cycler

AQUI=os.path.dirname(os.path.abspath(__file__)); RS=42; np.random.seed(RS)
GRANATE="#8B1538"; AZUL="#2c3e50"; AZUL2="#3498db"; CELESTE="#5b9bd5"
OK="#2e8b57"; MALO="#c0392b"; GRIS="#7f8c8d"; NARANJA="#e07b39"
plt.rcParams.update({"svg.fonttype":"path","figure.dpi":110,
    "axes.prop_cycle":cycler(color=[AZUL,GRANATE,AZUL2,OK,NARANJA,GRIS]),
    "axes.titlesize":15,"axes.titleweight":"bold","axes.titlecolor":AZUL,
    "axes.labelsize":12,"axes.edgecolor":"#c9ced3","axes.linewidth":1.1,
    "axes.grid":True,"grid.color":"#e9edf1","grid.linewidth":0.9,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False,"font.size":12,"font.family":"DejaVu Sans",
    "legend.frameon":False,"figure.autolayout":True})
def guardar(fig,n): fig.savefig(os.path.join(AQUI,n),format="svg",bbox_inches="tight",transparent=True); plt.close(fig); print(" +",n)

# superficie de desempeño (pico desplazado)
def _superficie():
    g=np.linspace(0,1,100); X,Y=np.meshgrid(g,g)
    Z=np.exp(-(((X-0.68)**2)/0.05+((Y-0.62)**2)/0.06))
    return X,Y,Z

# 1) Grid vs Random vs Bayesiana
def f_busqueda():
    X,Y,Z=_superficie()
    fig,axes=plt.subplots(1,3,figsize=(12.6,4.2))
    rng=np.random.default_rng(RS)
    # grid
    gx,gy=np.meshgrid(np.linspace(.1,.9,5),np.linspace(.1,.9,5))
    pts=[("GridSearch (25 puntos rígidos)",np.c_[gx.ravel(),gy.ravel()]),
         ("RandomSearch (20 al azar)",rng.uniform(.05,.95,(20,2))),
         ("Bayesiana / Optuna (20, busca el pico)",
          np.clip(np.r_[rng.uniform(.05,.95,(7,2)),
                        rng.normal([.68,.62],[.09,.09],(13,2))],0,1))]
    for ax,(t,P) in zip(axes,pts):
        ax.contourf(X,Y,Z,levels=12,cmap="Blues",alpha=.85)
        ax.scatter(P[:,0],P[:,1],c=GRANATE,s=42,edgecolor="white",zorder=3)
        ax.scatter([.68],[.62],marker="*",s=320,c=NARANJA,edgecolor="white",zorder=4)
        ax.set_title(t,loc="left",fontsize=12.5); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("hiperparámetro 1"); ax.set_ylabel("hiperparámetro 2")
    fig.suptitle("La búsqueda inteligente concentra los intentos donde el modelo rinde (★ óptimo)",
                 x=0.01,ha="left",fontsize=13.5,weight="bold",color=AZUL)
    guardar(fig,"fig_busqueda.svg")

# 2) Optimization history (Optuna)
def f_optuna_history():
    rng=np.random.default_rng(RS); n=60
    base=0.78+0.10*(1-np.exp(-np.arange(n)/12))
    vals=np.clip(base+rng.normal(0,0.012,n),0.74,0.9)
    best=np.maximum.accumulate(vals)
    fig,ax=plt.subplots(figsize=(8.6,4.6))
    ax.plot(range(1,n+1),vals,"o",color=CELESTE,alpha=.6,label="AUC por trial")
    ax.plot(range(1,n+1),best,color=GRANATE,lw=2.6,label="Mejor acumulado")
    ax.set_title("Historia de optimización: Optuna converge en ~50 trials",loc="left")
    ax.set_xlabel("Trial"); ax.set_ylabel("AUC (validación cruzada)"); ax.legend(loc="lower right")
    guardar(fig,"fig_optuna_history.svg")

# 3) Calibración (reliability diagram)
def f_calibracion():
    p=np.linspace(0.02,0.98,12)
    over = p**1.8                      # sobre-confiado (bajo la diagonal)
    under= p**0.55                     # conservador (sobre la diagonal)
    fig,ax=plt.subplots(figsize=(7.8,5.0))
    ax.plot([0,1],[0,1],"--",color=GRIS,lw=2,label="Perfectamente calibrado")
    ax.plot(p,under,"s-",color=AZUL2,lw=2,label="Conservador (subestima)")
    ax.plot(p,over,"o-",color=GRANATE,lw=2,label="Sobre-confiado (sobreestima)")
    ax.fill_between([0,1],[0,1],[0,0],color=MALO,alpha=.04)
    ax.set_title("Diagrama de calibración (reliability)",loc="left")
    ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Fracción real de positivos")
    ax.legend(loc="upper left"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    guardar(fig,"fig_calibracion.svg")

# 4) Profit curve
def f_profit_curve():
    t=np.linspace(0,1,200)
    profit=(-(t-0.27)**2)*900000+120000
    profit=profit-profit.min()*0.0
    fig,ax=plt.subplots(figsize=(8.6,4.8))
    ax.plot(t,profit/1000,color=AZUL,lw=3)
    bi=int(np.argmax(profit)); bt=t[bi]
    ax.axvline(bt,ls="--",color=OK,lw=2); ax.scatter([bt],[profit[bi]/1000],s=90,color=OK,zorder=5,edgecolor="white")
    ax.annotate(f"Óptimo: umbral={bt:.2f}\nprofit=${profit[bi]/1000:.0f}k",xy=(bt,profit[bi]/1000),
                xytext=(bt+0.12,profit[bi]/1000-30),color=OK,fontsize=11,
                arrowprops=dict(arrowstyle="->",color=OK))
    i05=np.argmin(abs(t-0.5))
    ax.axvline(0.5,ls=":",color=GRANATE,lw=1.6); ax.scatter([0.5],[profit[i05]/1000],s=70,color=GRANATE,zorder=5)
    ax.annotate(f"Umbral 0.5\n(por defecto)\n${profit[i05]/1000:.0f}k",xy=(0.5,profit[i05]/1000),
                xytext=(0.52,profit[i05]/1000-55),color=GRANATE,fontsize=10.5)
    ax.set_title("Profit curve: el umbral óptimo no es 0.5",loc="left")
    ax.set_xlabel("Umbral de decisión"); ax.set_ylabel("Profit (miles US$)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f"{v:,.0f}"))
    guardar(fig,"fig_profit_curve.svg")

if __name__=="__main__":
    print("Figuras S03 en",AQUI)
    f_busqueda(); f_optuna_history(); f_calibracion(); f_profit_curve()
    print("Listo.")

# -*- coding: utf-8 -*-
"""
Genera las figuras de la Sesión 01 (S01) como SVG vectorial, con una paleta
ÚNICA anclada en la identidad UNI del deck. Reemplazan a los antiguos diagramas
ASCII. Ejecutar con el venv del curso:

    python build_figuras.py

Salida: slides/assets/fig_*.svg
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle, FancyArrowPatch
from cycler import cycler

AQUI = os.path.dirname(os.path.abspath(__file__))
RS = 42
np.random.seed(RS)

# ===== Paleta única (identidad UNI + datos) =====
GRANATE = "#8B1538"; AZUL = "#2c3e50"; AZUL2 = "#3498db"; CELESTE = "#5b9bd5"
OK = "#2e8b57"; MALO = "#c0392b"; GRIS = "#7f8c8d"; NARANJA = "#e07b39"
PALETA = [AZUL, GRANATE, AZUL2, OK, NARANJA, GRIS]

plt.rcParams.update({
    "svg.fonttype": "path",          # texto como vectores -> idéntico en cualquier visor
    "figure.dpi": 110,
    "axes.prop_cycle": cycler(color=PALETA),
    "axes.titlesize": 16, "axes.titleweight": "bold", "axes.titlecolor": AZUL,
    "axes.labelsize": 13, "axes.edgecolor": "#c9ced3", "axes.linewidth": 1.1,
    "axes.grid": True, "grid.color": "#e9edf1", "grid.linewidth": 0.9,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 12.5, "font.family": "DejaVu Sans",
    "legend.frameon": False, "figure.autolayout": True,
})

def guardar(fig, nombre):
    ruta = os.path.join(AQUI, nombre)
    fig.savefig(ruta, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(" +", nombre)

def estilo_titulo(ax, titulo, sub=None):
    ax.set_title(titulo, loc="left", pad=(22 if sub else 10))
    if sub: ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9, color=GRIS, va="bottom")
    return ax

# ---------------------------------------------------------------- 1. Dispersión + recta OLS
def f_dispersion_recta():
    x = np.random.uniform(40, 220, 60)
    y = 50_000 + 1_200*x + np.random.normal(0, 14_000, x.size)
    b1, b0 = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.scatter(x, y/1000, s=44, color=AZUL2, alpha=.75, edgecolor="white", linewidth=.6, zorder=3, label="Viviendas")
    ax.plot(xs, (b0 + b1*xs)/1000, color=GRANATE, lw=3, zorder=4, label="Línea de mejor ajuste (OLS)")
    ax.set_xlabel("Superficie (m²)"); ax.set_ylabel("Precio (miles de US$)")
    ax.set_title("Regresión lineal: la recta que mejor pasa entre los puntos", loc="left")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:,.0f}"))
    guardar(fig, "fig_dispersion_recta.svg")

# ---------------------------------------------------------------- 2. Errores OLS (residuos)
def f_ols_errores():
    x = np.array([1,2,3,4,5,6,7,8.]);
    y = np.array([3,4.5,4,6,5.5,7.5,7,9.])
    b1,b0 = np.polyfit(x,y,1); yhat = b0+b1*x
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for xi,yi,yh in zip(x,y,yhat):
        ax.plot([xi,xi],[yi,yh], color=MALO, lw=2, zorder=2)
    ax.scatter(x,y, s=70, color=AZUL2, zorder=3, edgecolor="white", label="Valor real $y_i$")
    ax.plot(x, yhat, color=GRANATE, lw=2.8, zorder=2, label=r"Predicción $\hat{y}_i$")
    ax.scatter([],[], color=MALO, marker="|", s=120, label="Error (residuo)")
    ax.set_title("Mínimos cuadrados: minimizar la suma de errores al cuadrado", loc="left")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(loc="upper left")
    guardar(fig, "fig_ols_errores.svg")

# ---------------------------------------------------------------- 3. Gradiente descendente
def f_gradiente():
    beta = np.linspace(-1, 7, 200)
    L = (beta-3)**2 + 1
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(beta, L, color=AZUL, lw=3, zorder=2)
    # pasos del descenso
    b = 6.3; lr = 0.22; pts=[b]
    for _ in range(7):
        b = b - lr*2*(b-3); pts.append(b)
    pts = np.array(pts); Ls = (pts-3)**2+1
    ax.plot(pts, Ls, "o-", color=GRANATE, lw=1.6, ms=9, zorder=4, mec="white")
    ax.annotate("Inicio\n(pesos al azar)", xy=(pts[0],Ls[0]), xytext=(pts[0]-.2, Ls[0]+2.5),
                fontsize=11, color=GRANATE, ha="center")
    ax.annotate("Mínimo\n(mejor modelo)", xy=(3,1), xytext=(3,4.2), fontsize=11, color=OK, ha="center",
                arrowprops=dict(arrowstyle="->", color=OK, lw=1.6))
    ax.set_title("Gradiente descendente: pasitos cuesta abajo hacia el menor error", loc="left")
    ax.set_xlabel(r"Parámetro $\beta$"); ax.set_ylabel("Error (pérdida)")
    ax.set_yticks([])
    guardar(fig, "fig_gradiente.svg")

# ---------------------------------------------------------------- 4. Sigmoide
def f_sigmoide():
    z = np.linspace(-8, 8, 300); s = 1/(1+np.exp(-z))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(z, s, color=GRANATE, lw=3, zorder=3)
    ax.axhline(.5, color=GRIS, ls="--", lw=1); ax.axvline(0, color=GRIS, ls="--", lw=1)
    ax.scatter([0],[.5], s=80, color=AZUL, zorder=4, edgecolor="white")
    ax.annotate("z = 0  →  P = 0,5\n(umbral por defecto)", xy=(0,.5), xytext=(1.2,.28), fontsize=11, color=AZUL)
    ax.annotate("P → 1\n(evento casi seguro)", xy=(6,.99), xytext=(2.4,.74), fontsize=10.5, color=OK)
    ax.annotate("P → 0\n(muy improbable)", xy=(-6,.01), xytext=(-7.6,.20), fontsize=10.5, color=MALO)
    ax.set_title("Función sigmoide: de la recta a una probabilidad en [0, 1]", loc="left")
    ax.set_xlabel("z  =  β₀ + β₁x₁ + … + βₙxₙ"); ax.set_ylabel("P(Y = 1)"); ax.set_ylim(-.05,1.08)
    guardar(fig, "fig_sigmoide.svg")

# ---------------------------------------------------------------- 5. Log-odds
def f_logodds():
    p = np.linspace(.001,.999,300); logit = np.log(p/(1-p))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(p, logit, color=AZUL, lw=3)
    ax.axhline(0, color=GRIS, lw=1); ax.axvline(.5, color=GRIS, ls="--", lw=1)
    for pp in (.1,.5,.9):
        ax.scatter([pp],[np.log(pp/(1-pp))], s=70, color=GRANATE, zorder=4, edgecolor="white")
    ax.set_title("Log-odds (logit): convierte la probabilidad a una escala lineal (−∞, +∞)", loc="left")
    ax.set_xlabel("Probabilidad  P"); ax.set_ylabel("logit(P) = ln(P / (1−P))")
    guardar(fig, "fig_logodds.svg")

# ---------------------------------------------------------------- 6. Sesgo-Varianza
def f_bias_variance():
    c = np.linspace(1, 10, 100)
    train = 0.9*np.exp(-0.45*c) + 0.03
    val   = 0.9*np.exp(-0.45*c) + 0.03 + 0.011*(c-3.2)**2
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    ax.plot(c, train, color=AZUL2, lw=3, label="Error en entrenamiento")
    ax.plot(c, val,   color=GRANATE, lw=3, label="Error en validación")
    opt = c[np.argmin(val)]
    ax.axvline(opt, color=OK, ls="--", lw=1.6); ax.scatter([opt],[val.min()], s=80, color=OK, zorder=5, edgecolor="white")
    ax.axvspan(1, 3.4, color=AZUL2, alpha=.07); ax.axvspan(7.0, 10, color=MALO, alpha=.07)
    ax.text(2.1, val.max()*.92, "Subajuste\n(alto sesgo)", ha="center", color=AZUL2, fontsize=11, weight="bold")
    ax.text(8.5, val.max()*.92, "Sobreajuste\n(alta varianza)", ha="center", color=MALO, fontsize=11, weight="bold")
    ax.annotate("Punto óptimo", xy=(opt,val.min()), xytext=(opt+.4, val.min()+.12),
                color=OK, fontsize=11, arrowprops=dict(arrowstyle="->", color=OK))
    ax.set_title("Compromiso sesgo–varianza", loc="left")
    ax.set_xlabel("Complejidad del modelo →"); ax.set_ylabel("Error"); ax.set_yticks([]); ax.legend(loc="upper center")
    guardar(fig, "fig_bias_variance.svg")

# ---------------------------------------------------------------- 7. K-Fold CV
def f_cv_folds():
    k = 5
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    for i in range(k):
        for j in range(k):
            color = NARANJA if j==i else AZUL2
            ax.add_patch(Rectangle((j, k-1-i), .94, .82, color=color, ec="white", lw=2))
        ax.text(-0.25, k-1-i+.4, f"Iter {i+1}", ha="right", va="center", fontsize=11, color=AZUL)
    ax.add_patch(Rectangle((k+0.3, 3.2), .5, .82, color=AZUL2, ec="white")); ax.text(k+0.95, 3.6, "Entrenamiento", va="center", fontsize=11)
    ax.add_patch(Rectangle((k+0.3, 2.1), .5, .82, color=NARANJA, ec="white")); ax.text(k+0.95, 2.5, "Validación", va="center", fontsize=11)
    ax.set_xlim(-1.4, k+3.2); ax.set_ylim(-0.3, k+0.2); ax.axis("off")
    ax.set_title("Validación cruzada (k = 5): cada bloque es validación una vez", loc="left", color=AZUL)
    guardar(fig, "fig_cv_folds.svg")

# ---------------------------------------------------------------- 8. ROC + PR
def f_roc_pr():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    n=4000
    score = np.concatenate([np.random.normal(-1,1.1,int(n*.78)), np.random.normal(1.1,1.1,int(n*.22))])
    yt = np.r_[np.zeros(int(n*.78)), np.ones(int(n*.22))]
    p = 1/(1+np.exp(-score))
    fpr,tpr,_ = roc_curve(yt,p); a = auc(fpr,tpr)
    prec,rec,_ = precision_recall_curve(yt,p)
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(9.4,4.3))
    ax1.plot(fpr,tpr,color=GRANATE,lw=3,label=f"Modelo (AUC={a:.2f})"); ax1.plot([0,1],[0,1],"--",color=GRIS,label="Azar")
    ax1.set_title("Curva ROC", loc="left"); ax1.set_xlabel("FPR (1−especificidad)"); ax1.set_ylabel("TPR (sensibilidad)"); ax1.legend(loc="lower right")
    ax2.plot(rec,prec,color=AZUL,lw=3); ax2.axhline(yt.mean(), ls="--", color=GRIS, label=f"Base ({yt.mean():.0%})")
    ax2.set_title("Precision–Recall", loc="left"); ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.legend(loc="upper right")
    guardar(fig, "fig_roc_pr.svg")

# ---------------------------------------------------------------- 9. EDA distribuciones
def f_eda():
    rng = np.random.default_rng(RS)
    ingreso = rng.lognormal(8.4,.5,3000)
    edad = rng.normal(41,12,3000).clip(18,80)
    fig, axes = plt.subplots(1,3, figsize=(11.5,3.8))
    axes[0].hist(edad, bins=30, color=AZUL2, edgecolor="white"); axes[0].set_title("Histograma: edad", loc="left"); axes[0].set_xlabel("Años")
    axes[1].hist(ingreso, bins=40, color=GRANATE, edgecolor="white"); axes[1].set_title("Sesgo a la derecha: ingreso", loc="left"); axes[1].set_xlabel("US$")
    bp = axes[2].boxplot([edad, rng.normal(50,16,3000)], vert=True, patch_artist=True, labels=["Buenos","Morosos"])
    for patch,c in zip(bp["boxes"], [AZUL2, NARANJA]): patch.set_facecolor(c); patch.set_alpha(.75)
    axes[2].set_title("Boxplot por clase: edad", loc="left")
    fig.suptitle("EDA: conocer los datos antes de modelar", x=0.01, ha="left", fontsize=15, weight="bold", color=AZUL)
    guardar(fig, "fig_eda.svg")

# ---------------------------------------------------------------- 10. Correlación
def f_correlacion():
    rng = np.random.default_rng(RS)
    cols = ["ingreso","deuda","ratio_DI","edad","antigüedad","n_atrasos"]
    base = rng.normal(size=(1500, len(cols)))
    base[:,2] = .8*base[:,1] - .5*base[:,0] + .3*rng.normal(size=1500)   # ratio correlado
    base[:,4] = .6*base[:,3] + .4*rng.normal(size=1500)
    corr = np.corrcoef(base, rowvar=False)
    fig, ax = plt.subplots(figsize=(6.6,5.4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=10)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j,i,f"{corr[i,j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(corr[i,j])>.55 else "#333")
    ax.set_title("Mapa de correlación (detectar multicolinealidad)", loc="left", color=AZUL)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    guardar(fig, "fig_correlacion.svg")

# ---------------------------------------------------------------- 11. Interpretación de coeficientes
def f_coef_interpret():
    feats=["antigüedad (por año)","metros² (por m²)","habitación (c/u)"]
    coefs=[-1200,1250,8500]
    fig,ax=plt.subplots(figsize=(8.6,4.0))
    ax.barh(feats,coefs,color=[OK if c>0 else MALO for c in coefs])
    ax.axvline(0,color="#333",lw=1)
    for f,c in zip(feats,coefs):
        ax.text(c+(180 if c>0 else -180),f,f"{c:+,} US$",va="center",ha="left" if c>0 else "right",fontsize=11.5,weight="bold",
                color=OK if c>0 else MALO)
    ax.set_xlim(-3000,11000)
    estilo_titulo(ax,"Interpretación de coeficientes: cuánto suma o resta cada variable al precio","verde = sube el precio · rojo = lo baja")
    ax.set_xlabel("Efecto sobre el precio (US$)")
    guardar(fig,"fig_coef_interpret.svg")

# ---------------------------------------------------------------- 12. Odds Ratio forest (credit scoring)
def f_odds_forest():
    variables=["Edad","Tiene hipoteca","Meses último atraso","Deuda / Ingreso"]
    OR=[0.98,0.67,1.08,4.48]
    fig,ax=plt.subplots(figsize=(8.8,4.2))
    cols=[OK if o<1 else MALO for o in OR]
    ax.hlines(variables,[1]*4,OR,color=cols,lw=3,zorder=2)
    ax.scatter(OR,variables,color=cols,s=110,zorder=3,edgecolor="white")
    ax.axvline(1,color=GRIS,ls="--",lw=1.6)
    for v,o in zip(variables,OR):
        ax.text(o*(1.08 if o>=1 else 0.92),v,f"OR={o}",va="center",ha="left" if o>=1 else "right",fontsize=11,weight="bold",color=OK if o<1 else MALO)
    ax.set_xscale("log"); ax.set_xlim(0.5,7)
    ax.text(0.62,3.4,"◄ Protege",color=OK,fontsize=11,weight="bold")
    ax.text(3.2,0.4,"Aumenta el riesgo ►",color=MALO,fontsize=11,weight="bold",ha="right")
    estilo_titulo(ax,"Odds Ratio por variable (credit scoring)","OR>1 aumenta la probabilidad de default · OR<1 la reduce")
    ax.set_xlabel("Odds Ratio (escala logarítmica)")
    guardar(fig,"fig_odds_forest.svg")

# ---------------------------------------------------------------- 13. MAE vs RMSE y outliers
def f_mae_rmse():
    rng=np.random.default_rng(0); err=np.abs(rng.normal(0,5,40))
    out=np.append(err,60.)
    mae1,rmse1=err.mean(),np.sqrt((err**2).mean())
    mae2,rmse2=out.mean(),np.sqrt((out**2).mean())
    fig,ax=plt.subplots(figsize=(8.4,4.4))
    ax.bar([-.19,.81],[mae1,mae2],width=.38,color=AZUL2,label="MAE")
    ax.bar([.19,1.19],[rmse1,rmse2],width=.38,color=GRANATE,label="RMSE")
    for x,v in [(-.19,mae1),(.81,mae2),(.19,rmse1),(1.19,rmse2)]:
        ax.text(x,v+0.3,f"{v:.1f}",ha="center",fontsize=10.5,weight="bold")
    ax.set_xticks([0,1]); ax.set_xticklabels(["Sin outlier","Con 1 outlier (60)"])
    estilo_titulo(ax,"RMSE castiga los outliers mucho más que MAE","un solo error grande dispara el RMSE")
    ax.set_ylabel("Error"); ax.legend(loc="upper left")
    guardar(fig,"fig_mae_rmse.svg")

# ---------------------------------------------------------------- 14. Regularización Ridge/Lasso
def f_regularizacion():
    feats=[f"x{i}" for i in range(1,9)]
    orig=np.array([2.1,-1.8,0.9,1.5,-0.4,0.25,-1.1,0.7])
    ridge=orig*0.5
    lasso=orig.copy(); lasso[np.abs(orig)<0.6]=0.; lasso*=0.78
    fig,axes=plt.subplots(1,3,figsize=(12.4,3.8),sharey=True)
    for ax,(t,c,col) in zip(axes,[("Sin regularizar",orig,AZUL),("Ridge (L2): encoge todo",ridge,AZUL2),("Lasso (L1): lleva a cero",lasso,GRANATE)]):
        ax.bar(feats,c,color=col); ax.axhline(0,color="#333",lw=.8); ax.set_title(t,loc="left",fontsize=12.5)
    axes[2].annotate("= 0\n(descartadas)",xy=(4,0),xytext=(4.4,1.1),color=GRANATE,fontsize=10,ha="center",arrowprops=dict(arrowstyle="->",color=GRANATE))
    fig.suptitle("Regularización: Ridge encoge los coeficientes; Lasso lleva algunos a CERO (selección automática)",x=0.01,ha="left",fontsize=13,weight="bold",color=AZUL)
    guardar(fig,"fig_regularizacion.svg")

# ---------------------------------------------------------------- 15. Data leakage: caída en producción
def f_leakage_drop():
    fig,ax=plt.subplots(figsize=(8.2,4.4))
    etapas=["Train\n(con fuga)","Test\n(con fuga)","Producción\n(datos reales)"]
    acc=[99,98,62]; cols=[AZUL2,AZUL2,MALO]
    b=ax.bar(etapas,acc,color=cols)
    for r,v in zip(b,acc): ax.text(r.get_x()+r.get_width()/2,v+1.5,f"{v}%",ha="center",fontsize=12,weight="bold")
    ax.set_ylim(0,108)
    ax.annotate("¡el modelo colapsa\nen el mundo real!",xy=(2,62),xytext=(0.9,38),color=MALO,fontsize=11,weight="bold",
                arrowprops=dict(arrowstyle="->",color=MALO,lw=1.6))
    estilo_titulo(ax,"Data leakage: accuracy 'perfecto' en test, desastre en producción")
    ax.set_ylabel("Accuracy (%)")
    guardar(fig,"fig_leakage_drop.svg")

if __name__ == "__main__":
    print("Generando figuras de S01 en", AQUI)
    f_dispersion_recta(); f_ols_errores(); f_gradiente(); f_sigmoide(); f_logodds()
    f_bias_variance(); f_cv_folds(); f_roc_pr(); f_eda(); f_correlacion()
    f_coef_interpret(); f_odds_forest(); f_mae_rmse(); f_regularizacion(); f_leakage_drop()
    print("Listo.")

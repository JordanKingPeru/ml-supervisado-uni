import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix(y_true, y_pred, labels=None, title='Matriz de Confusión'):
    """
    Grafica una matriz de confusión estilizada.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels if labels else "auto",
                yticklabels=labels if labels else "auto")
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title(title)
    plt.show()


def plot_roc_curve(y_true, y_prob, title='Curva ROC'):
    """
    Grafica la curva ROC y calcula el AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Grafica la importancia de variables para modelos basados en árboles.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 6))
        plt.title(f"Top {top_n} Feature Importances")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), [feature_names[i]
                   for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("El modelo no tiene atributo 'feature_importances_'")


def load_data(filename, data_path='data', **kwargs):
    """
    Intenta cargar datos desde diferentes ubicaciones comunes.
    Busca en rutas relativas estándar para notebooks y scripts.
    Acepta argumentos adicionales para pd.read_csv (ej. sep=';').
    """
    # Datos centralizados del curso: <raíz>/03_Datos/tabular/ (anclado a este
    # archivo, funciona sin importar desde dónde se ejecute el notebook)
    from pathlib import Path
    datos_curso = Path(__file__).resolve().parent.parent.parent / "03_Datos" / "tabular"

    # Posibles rutas relativas desde donde se ejecuta el notebook
    paths = [
        str(datos_curso / filename),         # 03_Datos/tabular/ (centralizado)
        # Ejecutando desde root o carpeta correcta
        f"{data_path}/{filename}",
        f"../{data_path}/{filename}",        # Ejecutando desde notebooks/
        f"../../{data_path}/{filename}",     # Ejecutando desde más profundo
        filename                             # En el mismo directorio
    ]

    for path in paths:
        try:
            return pd.read_csv(path, **kwargs)
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        f"No se pudo encontrar {filename} en ninguna de las rutas esperadas: {paths}")

# 🎓 Machine Learning Supervisado

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

## Programa de Especialización en Ciencia de Datos (PECD)
### Universidad Nacional de Ingeniería (UNI)

---

## 👨‍🏫 Docente
**Cand. Ph.D Jordan Rodriguez Mallqui**
- 📧 jrodriguezm216@gmail.com
- 🌐 [jordandataexpert.com](https://jordandataexpert.com)
- 💼 [LinkedIn](https://linkedin.com/in/jordan-rodriguez-peru)
- 💻 [GitHub](https://github.com/JordanKingPeru)

---

## 📚 Contenido del curso

| Sesión | Tema | Módulo |
|--------|------|--------|
| 00 | Instalación del entorno (antes de la sesión 1) | `02_Contenido/M00_Onboarding/` |
| 01 | Pipelines, regresión lineal/logística, data leakage | `02_Contenido/M01_Pipelines_Baselines/` |
| 02 | Árboles, Random Forest, XGBoost, SVM, KNN | `02_Contenido/M02_Arboles_Ensembles/` |
| 03 | Optimización (Optuna), calibración, métricas de negocio | `02_Contenido/M03_Optimizacion_Negocio/` |
| 04 | SHAP, validación avanzada, despliegue (Streamlit/Docker) | `02_Contenido/M04_XAI_Produccion/` |
| 🎁 | Bonus: conexión a bases de datos y repositorios públicos | `02_Contenido/M06_Conexion_Bases_Datos/` |
| 🎁 | Bonus de cierre: caso integrador end-to-end como **documento regulatorio** de riesgo de crédito (dataset real UCI, scorecard + LightGBM, HTML/PDF) | `02_Contenido/M07_Caso_Integrador_ML/` |

Además:
- `02_Contenido/Ejercicios_practica.md` — ejercicios guiados por módulo + **proyecto final**.
- `02_Contenido/FAQ_Preguntas_Respuestas.md` — preguntas frecuentes del curso.
- `03_Datos/` — datasets del curso (los notebooks los cargan automáticamente).
- `Silabo_ML_Supervisado_PECD.pdf` — sílabo oficial.

📖 **Cómo empezar**: lee [LEEME_Alumnos.md](LEEME_Alumnos.md).

---

## 🛠️ Instalación rápida

```bash
# Clonar repositorio
git clone https://github.com/JordanKingPeru/ml-supervisado-uni.git
cd ml-supervisado-uni

# Crear entorno e instalar dependencias
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/Mac: source .venv/bin/activate
pip install -r 02_Contenido/M00_Onboarding/requirements.txt
```

Guía completa (incl. conda y verificación del entorno): `02_Contenido/M00_Onboarding/guia_instalacion.md`.

> 💡 También puedes abrir este repo en **GitHub Codespaces** (botón "Code → Codespaces"): el
> devcontainer instala todo y levanta la app demo de la sesión 4 automáticamente.

---

## 📂 Estructura de cada módulo

```
Mxx_Tema/
├── notebooks/   ← notebooks numerados en orden de clase (los *_Alumnos son para completar)
├── slides/      ← diapositivas en PDF
└── README.md    ← objetivos del módulo
```

`02_Contenido/recursos/` contiene utilidades compartidas (`utils.py`, `config.py`) que los notebooks
importan — no muevas esa carpeta ni `03_Datos/`.

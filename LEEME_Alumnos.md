# Machine Learning Supervisado — Material del alumno

¡Bienvenido/a! Puedes usar este material de tres formas, de menor a mayor esfuerzo:

## Opción A — GitHub Codespaces (sin instalar nada)
1. En la página del repo: **Code → Codespaces → Create codespace**.
2. Espera a que termine la instalación automática (unos minutos la primera vez).
3. Abre cualquier notebook de `02_Contenido/Mxx_*/notebooks/` y ejecútalo. La app demo de la
   sesión 4 se levanta sola en el puerto 8501.

## Opción B — Google Colab (solo necesitas navegador)
1. Sube el notebook que quieras a [colab.research.google.com](https://colab.research.google.com).
2. Sube también `03_Datos/tabular/` y la carpeta `02_Contenido/recursos/` (o clona el repo dentro de
   Colab: `!git clone https://github.com/JordanKingPeru/ml-supervisado-uni.git`).
3. Instala lo que falte: `!pip install -r ml-supervisado-uni/02_Contenido/M00_Onboarding/requirements.txt`.

## Opción C — Tu máquina (recomendada para el curso)
1. Instala **Python 3.10+** ([python.org](https://www.python.org/downloads/)) y
   **VS Code** con la extensión de Python (o Jupyter Lab).
2. Clona el repo y crea el entorno:
   ```bash
   git clone https://github.com/JordanKingPeru/ml-supervisado-uni.git
   cd ml-supervisado-uni
   python -m venv .venv
   # Windows: .venv\Scripts\activate     Linux/Mac: source .venv/bin/activate
   pip install -r 02_Contenido/M00_Onboarding/requirements.txt
   ```
3. Verifica tu instalación con `02_Contenido/M00_Onboarding/checklist_verificacion.md`.
4. Abre los notebooks en orden (cada módulo tiene su `README.md` con los objetivos).

## Reglas de oro
- **No muevas** `02_Contenido/recursos/` ni `03_Datos/`: los notebooks los encuentran con rutas
  relativas (vía `utils.load_data()`).
- Los notebooks `*_Alumnos.ipynb` son **versiones para completar en clase** — tienen celdas con
  `# TU CÓDIGO AQUÍ` que fallan a propósito hasta que las resuelvas.
- Tras cada sesión, haz los ejercicios de `02_Contenido/Ejercicios_practica.md` (30–60 min).
- ¿Dudas? Revisa primero `02_Contenido/FAQ_Preguntas_Respuestas.md`.
- Los libros recomendados están en `Referencias_libros.md` (con enlaces oficiales).
- 🎁 **Cierre del curso:** `02_Contenido/M07_Caso_Integrador_ML/` es un caso end-to-end completo
  resuelto como **documento regulatorio** de riesgo de crédito (dataset real de UCI). Mira el resultado
  final en `salidas/` (HTML y PDF) y úsalo como modelo para tu proyecto final.

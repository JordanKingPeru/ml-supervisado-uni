# 🛠️ Guía de Instalación del Entorno "Data Science Pro"
**Docente**: Jordan King Rodriguez Mallqui 

Esta guía te ayudará a configurar un entorno de desarrollo profesional para Machine Learning, idéntico al utilizado en la industria.

## 📋 Prerrequisitos

1.  **Sistema Operativo:** Windows 10/11, macOS o Linux.
2.  **RAM:** Mínimo 8GB recomendado.
3.  **Permisos:** Acceso de administrador para instalar software.

---

## 🚀 Paso 1: Visual Studio Code (El Editor)

Olvidemos Jupyter Notebook en el navegador por un momento. Usaremos **Visual Studio Code**, el estándar de la industria.

1.  **Descargar e Instalar:**
    *   Ve a [code.visualstudio.com](https://code.visualstudio.com/) y descarga la versión para tu sistema operativo.
    *   Instala con las opciones por defecto.

2.  **Extensiones OBLIGATORIAS:**
    Abre VS Code, ve a la pestaña de extensiones (cuadrados a la izquierda) e instala:
    *   **Python** (Microsoft)
    *   **Jupyter** (Microsoft)
    *   **Pylance** (Microsoft)
    *   **GitLens** (Opcional, pero recomendada para ver quién modificó qué)

---

## 🐍 Paso 2: Entorno Virtual (Anaconda / Miniconda)

Para evitar conflictos de versiones ("Dependency Hell"), crearemos un ambiente aislado.

1.  **Instalar Miniconda (Recomendado):**
    *   Descarga el instalador desde [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html).
    *   Durante la instalación, marca la opción "Add Miniconda3 to my PATH environment variable" (si estás en Windows y sabes lo que haces) o usa "Anaconda Prompt" después.

2.  **Crear el Entorno:**
    Abre tu terminal (o Anaconda Prompt) y navega hasta la carpeta donde descargaste los archivos del curso (donde está `environment.yml`).

    Ejecuta:
    ```bash
    conda env create -f environment.yml
    ```

    *Alternativa manual:*
    ```bash
    conda create -n ml_pro python=3.10 -y
    conda activate ml_pro
    pip install -r requirements.txt
    ```

3.  **Activar el Entorno:**
    ```bash
    conda activate ml_pro
    ```

---

## 🐙 Paso 3: Git & GitHub (Tu Portafolio)

En la industria, **si no está en Git, no existe**.

1.  **Cuenta de GitHub:**
    *   Crea una cuenta en [GitHub.com](https://github.com/) si no tienes una.

2.  **Instalar Git:**
    *   Windows: [git-scm.com/download/win](https://git-scm.com/download/win)
    *   Mac: `brew install git` (si usas Homebrew) o descarga desde la web.

3.  **Configuración Inicial:**
    Abre tu terminal y configura tu identidad:
    ```bash
    git config --global user.name "Tu Nombre"
    git config --global user.email "tu@email.com"
    ```

4.  **Clonar/Crear Repositorio:**
    *   Crea una carpeta para el curso.
    *   Inicializa el repo: `git init`
    *   O clona el repo del curso (si aplica).

---

## ✅ Verificación

Para asegurar que todo está listo, ejecuta el siguiente comando en tu terminal con el entorno activado:

```bash
python -c "import pandas; import sklearn; import xgboost; print('¡Entorno ML Pro configurado correctamente!')"
```

Si ves el mensaje de éxito, ¡estás listo para empezar!

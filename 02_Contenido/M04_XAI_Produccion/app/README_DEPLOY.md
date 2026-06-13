# 🚀 Guía de Despliegue - Credit Scoring App

> **Curso:** Machine Learning Supervisado - PECD UNI  
> **Sesión:** 04 - De la Caja Negra a la Realidad Productiva

Esta guía explica cómo configurar y desplegar la aplicación de Credit Scoring con Streamlit.

---

## 📋 Tabla de Contenidos

1. [Requisitos Previos](#1-requisitos-previos)
2. [Estructura del Proyecto](#2-estructura-del-proyecto)
3. [Instalación Local](#3-instalación-local)
4. [Ejecución Local](#4-ejecución-local)
5. [Configuración de la App](#5-configuración-de-la-app)
6. [Despliegue en Streamlit Cloud](#6-despliegue-en-streamlit-cloud)
7. [Despliegue con Docker](#7-despliegue-con-docker)
8. [Solución de Problemas](#8-solución-de-problemas)

---

## 1. Requisitos Previos

### Software Necesario
- **Python:** 3.9 o superior
- **pip:** Última versión
- **Git:** Para control de versiones (opcional)

### Verificar Instalación
```bash
python --version   # Debería mostrar Python 3.9+
pip --version      # pip 21.0+
```

---

## 2. Estructura del Proyecto

```
app/
├── app.py                 # 🎯 Aplicación principal de Streamlit
├── requirements.txt       # 📦 Dependencias de Python
├── README_DEPLOY.md       # 📖 Esta guía
└── models/
    ├── model_joblib.joblib    # 🤖 Modelo serializado
    ├── model_lgb.txt          # 🌳 Modelo en formato nativo
    └── model_metadata.json    # 📋 Metadatos del modelo
```

### Archivos Clave

| Archivo | Descripción |
|---------|-------------|
| `app.py` | Código principal de la interfaz Streamlit |
| `requirements.txt` | Lista de dependencias con versiones |
| `model_joblib.joblib` | Modelo LightGBM + metadatos (generado por notebook) |
| `model_metadata.json` | Información del modelo (AUC, features, fecha) |

---

## 3. Instalación Local

### Paso 1: Navegar a la carpeta de la app
```bash
cd "ruta/a/02_Contenido/M04_XAI_Produccion/app"
```

### Paso 2: Crear entorno virtual (recomendado)
```bash
# Crear entorno
python -m venv venv

# Activar entorno
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
```

### Paso 3: Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 4: Verificar que el modelo existe
```bash
# Debe mostrar los archivos del modelo
ls models/
# O en Windows:
dir models/
```

> **⚠️ Importante:** Si la carpeta `models/` está vacía, ejecuta primero el notebook `03_Model_Serialization.ipynb` para generar el modelo.

---

## 4. Ejecución Local

### Comando Básico
```bash
streamlit run app.py
```

### Con configuración personalizada
```bash
streamlit run app.py --server.port 8080 --server.address localhost
```

### Resultado Esperado
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Abre tu navegador en `http://localhost:8501` para ver la app.

---

## 5. Configuración de la App

### 5.1 Variables de Entrada

La app usa **5 variables simplificadas** para la demo:

| Variable | Descripción | Rango |
|----------|-------------|-------|
| `SD_MAX_DIAS_MORA_SSFF_06M` | Variabilidad de días de mora | 0 - 50 |
| `MAX_PORC_DEUDA_SOBREGIRO...` | % Máximo de sobregiro | 0% - 100% |
| `MAX_CNT_ENTIDADES_SSFF_06M` | Entidades consultadas | 0 - 15 |
| `NumeroTrabajadores` | Empleados de la empresa | 1 - 500 |
| `ANTIGUEDAD_RCC_01M` | Antigüedad en meses | 0 - 240 |

### 5.2 Umbrales de Decisión

Puedes modificar los umbrales en `app.py`:

```python
# Línea ~85 en app.py
THRESHOLD_LOW = 0.3   # Por debajo: Aprobar
THRESHOLD_HIGH = 0.6  # Por encima: Rechazar
```

| Rango de Probabilidad | Decisión |
|-----------------------|----------|
| 0% - 30% | ✅ APROBAR |
| 30% - 60% | ⚠️ REVISIÓN MANUAL |
| 60% - 100% | ❌ RECHAZAR |

### 5.3 Personalizar Features

Para agregar o modificar variables, edita el diccionario `FEATURE_CONFIG`:

```python
FEATURE_CONFIG = {
    'NOMBRE_VARIABLE': {
        'label': '📊 Etiqueta visible',
        'description': 'Descripción para el usuario',
        'min': 0.0, 
        'max': 100.0, 
        'default': 50.0, 
        'step': 1.0,
        'help': 'Tooltip de ayuda'
    },
    # ... más variables
}
```

---

## 6. Despliegue en Streamlit Cloud

### Paso 1: Subir a GitHub
```bash
git init
git add .
git commit -m "Credit Scoring App"
git remote add origin https://github.com/tu-usuario/credit-scoring-app.git
git push -u origin main
```

### Paso 2: Conectar con Streamlit Cloud
1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Haz clic en "New app"
3. Conecta tu repositorio de GitHub
4. Configura:
   - **Repository:** `tu-usuario/credit-scoring-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Haz clic en "Deploy"

### Paso 3: URL Pública
Tu app estará disponible en:
```
https://tu-usuario-credit-scoring-app-xxxx.streamlit.app
```

---

## 7. Despliegue con Docker

### 7.1 Crear Dockerfile

Crea un archivo `Dockerfile` en la carpeta `app/`:

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copiar requirements primero (para cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación
COPY . .

# Puerto de Streamlit
EXPOSE 8501

# Comando de inicio
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 7.2 Construir y Ejecutar

```bash
# Construir imagen
docker build -t credit-scoring-app .

# Ejecutar contenedor
docker run -p 8501:8501 credit-scoring-app
```

### 7.3 Docker Compose (Opcional)

Crea `docker-compose.yml`:

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

```bash
docker-compose up -d
```

---

## 8. Solución de Problemas

### ❌ Error: "No se encontró el archivo del modelo"

**Causa:** El modelo no ha sido generado.

**Solución:**
1. Ejecuta el notebook `03_Model_Serialization.ipynb`
2. Verifica que exista `models/model_joblib.joblib`

### ❌ Error: "ModuleNotFoundError: No module named 'plotly'"

**Causa:** Dependencias no instaladas.

**Solución:**
```bash
pip install -r requirements.txt
```

### ❌ La app no muestra gráficos

**Causa:** Versión antigua de Plotly.

**Solución:**
```bash
pip install --upgrade plotly
```

### ❌ Error de memoria en Streamlit Cloud

**Causa:** Modelo muy grande o dataset en memoria.

**Solución:**
- Usa `@st.cache_resource` para cachear el modelo
- Reduce el tamaño del modelo (menos estimadores)

### ❌ Puerto 8501 ocupado

**Solución:**
```bash
streamlit run app.py --server.port 8502
```

---

## 📚 Recursos Adicionales

- [Documentación Oficial de Streamlit](https://docs.streamlit.io/)
- [Streamlit Cloud](https://streamlit.io/cloud)
- [Galería de Apps Streamlit](https://streamlit.io/gallery)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

---

## 🎓 Créditos

**Curso:** Machine Learning Supervisado  
**Programa:** Especialización en Ciencia de Datos (PECD)  
**Universidad:** Universidad Nacional de Ingeniería (UNI)  
**Año:** (por edición)

---



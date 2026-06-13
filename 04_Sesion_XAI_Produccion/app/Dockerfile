# Imagen base de Python
FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Copiar requirements primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación
COPY . .

# Puerto que expone Streamlit
EXPOSE 8501

# Healthcheck para monitoreo
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando para ejecutar la app
ENTRYPOINT ["streamlit", "run", "app_basic.py", "--server.port=8501", "--server.address=0.0.0.0"]

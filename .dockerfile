# Usa Python 3.11 slim como base
FROM python:3.11-slim

# Variables de entorno para evitar archivos .pyc y loguear a stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Dependencias de sistema básicas
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copia todos los archivos del proyecto al contenedor
COPY . .

# Instala el proyecto de forma editable y pytest
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir pytest

# Comando por defecto: ejecutar los tests
CMD ["pytest", "-q"]

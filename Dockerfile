# Docker image for running the Clinic LLM Test Framework tests in CI.

FROM python:3.11-slim

# Avoid .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project into the image
COPY . .

# Install the package and pytest
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir pytest

# Default command: run the full test suite
CMD ["pytest", "-q"]

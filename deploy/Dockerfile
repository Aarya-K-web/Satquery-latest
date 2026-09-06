# SatQuery EvidenceSwarm (SIH26167) — CPU Deployment Container
# Optimized for Railway / Render / Cloud Run (CPU-only, no CUDA dependencies)
FROM python:3.11-slim

WORKDIR /app

# Install minimal OS dependencies for raster and geospatial I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install lightweight CPU dependencies
COPY requirements-cpu.txt .
RUN pip install --no-cache-dir -r requirements-cpu.txt

# Copy application source code, UI, and demo data
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY demo/ ./demo/
COPY docs/ ./docs/

# Create runtime persistent directories
RUN mkdir -p uploads data

ENV PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Launch FastAPI orchestrator
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

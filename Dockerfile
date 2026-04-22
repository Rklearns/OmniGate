FROM python:3.11-slim

# XGBoost relies on OpenMP at runtime, and build-essential helps when a package falls back to source builds.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY preprocessing ./preprocessing
COPY README.md .

# Keep matplotlib fully headless inside the container and direct caches to writable locations.
ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp

RUN mkdir -p /app/results_aggregated

CMD ["python", "-m", "src.main"]

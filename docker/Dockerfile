FROM python:3.10-slim

LABEL maintainer="Amarnath <amarnath@example.com>"
LABEL description="Nuclear Glass Corrosion Prediction"

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . --no-deps

RUN useradd -m appuser && chown -R appuser /workspace
USER appuser

CMD ["python", "-m", "nuclear_glass"]

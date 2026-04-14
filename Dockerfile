# Trading API (FastAPI + PyTorch CPU). Cloud Run sets PORT; default 8080.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
# CPU-only torch keeps the image smaller than the default CUDA wheel.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
# resolve_checkpoint() also checks project root; place weights here or set DQN_CHECKPOINT.
COPY Q_net_best.pt ./Q_net_best.pt

EXPOSE 8080
ENV PORT=8080

CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT}"]

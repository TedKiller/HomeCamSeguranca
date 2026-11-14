FROM python:3.11-slim

ARG ORT_PACKAGE=onnxruntime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOMECAM_RECORDINGS_DIR=/data/gravacoes \
    HOMECAM_CONFIG_PATH=/data/config.json \
    HOMECAM_WEB_HOST=0.0.0.0 \
    HOMECAM_WEB_PORT=8765

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# Em containers Linux substitui onnxruntime-directml (exclusivo de Windows) por onnxruntime/onnxruntime-gpu
RUN grep -vi "onnxruntime-directml" /app/requirements.txt > /tmp/req-linux.txt \
    && echo "${ORT_PACKAGE}>=1.18.0" >> /tmp/req-linux.txt \
    && pip install --no-cache-dir -r /tmp/req-linux.txt \
    && rm -f /tmp/req-linux.txt

COPY *.py /app/
COPY assets /app/assets

RUN mkdir -p /data/gravacoes /data/secrets

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8765/api/health || exit 1

CMD ["python", "main.py", "--headless", "--web-host", "0.0.0.0", "--web-port", "8765"]


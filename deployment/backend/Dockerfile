FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SAFETRACE_OFFLINE=1 \
    SAFETRACE_PROJECT_ROOT=/app \
    SAFETRACE_DATA_DIR=/var/lib/safetrace/data \
    SAFETRACE_ANALYSIS_SAFE_MODE=true \
    SAFETRACE_DEVICE=auto \
    SAFETRACE_VLM_ENABLED=0 \
    SAFETRACE_ENABLE_VLM=0 \
    SAFETRACE_CHAT_ENABLED=0

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 safetrace

WORKDIR /app

COPY deployment/backend/requirements-cloud.txt /tmp/requirements-cloud.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements-cloud.txt

COPY --chown=safetrace:safetrace src /app/src
COPY --chown=safetrace:safetrace config/safetrace.cloud.env.example /app/config/safetrace.cloud.env.example
COPY --chown=safetrace:safetrace checkpoints /app/checkpoints
COPY --chown=safetrace:safetrace deployment/backend/start.sh /app/start.sh
RUN chmod 0755 /app/start.sh && mkdir -p /var/lib/safetrace/data && chown -R safetrace:safetrace /app /var/lib/safetrace

USER safetrace
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/api/health', timeout=3)"
ENTRYPOINT ["/app/start.sh"]

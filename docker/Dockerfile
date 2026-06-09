# SafeTrace — GPU-enabled, offline-ready container
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SAFETRACE_OFFLINE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

# System libs needed by OpenCV / video decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first to maximize layer cache reuse.
COPY docker/requirements.txt /app/docker/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/docker/requirements.txt

# Project source
COPY src/        /app/src/
COPY frontend/   /app/frontend/
COPY main.py     /app/main.py
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN sed -i 's/\r$//' /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

# Checkpoints + data are mounted at runtime (`-v`) so the image stays slim.
RUN mkdir -p /app/checkpoints /app/data/frames

EXPOSE 8501

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["ui"]

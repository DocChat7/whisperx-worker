FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/root/.cache/torch
ENV HF_HOME=/root/.cache/huggingface

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install into system Python (torch already in base image)
RUN uv pip install --system --no-cache-dir \
    runpod \
    "whisperx==3.8.5"

# Pre-download Whisper large-v3 model (baked into image)
RUN python -c "\
import whisperx; \
whisperx.load_model('large-v3', 'cpu', compute_type='int8'); \
print('Whisper cached!')"

# Pre-download diarization model (needs HF_TOKEN at build time)
ARG HF_TOKEN=""
RUN if [ -n "$HF_TOKEN" ]; then \
    python -c "\
import os; os.environ['HF_TOKEN']='${HF_TOKEN}'; \
from whisperx.diarize import DiarizationPipeline; \
DiarizationPipeline(token='${HF_TOKEN}', device='cpu'); \
print('Diarization model cached!')"; \
    else echo "HF_TOKEN not provided, diarization model will download at runtime"; fi

# Cleanup
RUN rm -rf /root/.cache/pip /tmp/*

COPY handler.py .

CMD ["python", "-u", "handler.py"]

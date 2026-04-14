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

# Install into system Python (torch already in base image, no venv needed)
RUN uv pip install --system --no-cache-dir \
    runpod \
    "whisperx==3.8.5"

# Pre-download Whisper large-v3 model
RUN python -c "\
import whisperx; \
whisperx.load_model('large-v3', 'cpu', compute_type='int8'); \
print('Model cached!')"

# Cleanup
RUN rm -rf /root/.cache/pip /tmp/*

COPY handler.py .

CMD ["python", "-u", "handler.py"]

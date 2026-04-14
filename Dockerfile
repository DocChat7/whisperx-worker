FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/root/.cache/torch
ENV HF_HOME=/root/.cache/huggingface

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install deps from lock file (reproducible)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Pre-download Whisper large-v3 model
RUN uv run python3 -c "\
import torch; \
torch.serialization.add_safe_globals([ \
    __import__('omegaconf').listconfig.ListConfig, \
    __import__('omegaconf').dictconfig.DictConfig]); \
import whisperx; \
whisperx.load_model('large-v3', 'cpu', compute_type='int8'); \
print('Model cached!')"

# Pre-download DeepFilterNet model
RUN uv run python3 -c "from df.enhance import init_df; init_df(config_allow_defaults=True)" || true

# Cleanup
RUN rm -rf /root/.cache/pip /tmp/*

COPY handler.py .

CMD ["uv", "run", "python3", "-u", "handler.py"]

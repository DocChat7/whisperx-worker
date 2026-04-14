FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/root/.cache/torch
ENV HF_HOME=/root/.cache/huggingface

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Pin torch+torchaudio 2.6 (compatible with pyannote.audio 3.1)
RUN pip3 install --no-cache-dir \
    torch==2.6.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install whisperx + deps with compatible pyannote
RUN pip3 install --no-cache-dir \
    whisperx \
    deepfilternet \
    runpod \
    "pyannote.audio>=3.1,<3.3"

# Pre-download Whisper large-v3
RUN python3 -c "\
import torch; \
torch.serialization.add_safe_globals([ \
    __import__('omegaconf').listconfig.ListConfig, \
    __import__('omegaconf').dictconfig.DictConfig]); \
import whisperx; \
whisperx.load_model('large-v3', 'cpu', compute_type='int8'); \
print('Model cached!')"

# Pre-download DeepFilterNet
RUN python3 -c "from df.enhance import init_df; init_df(config_allow_defaults=True)" || true

# Cleanup
RUN rm -rf /root/.cache/pip /tmp/*

WORKDIR /app
COPY handler.py .

CMD ["python3", "-u", "handler.py"]

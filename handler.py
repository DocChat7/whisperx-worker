"""
RunPod Serverless Handler — WhisperX Transcription

Modelos pré-carregados na GPU. Configurável por request.
Denoise via FFmpeg (CPU, sem dependência extra).

Input:
    audio_base_64: str       — áudio em base64
    language: str            — idioma (default: "pt")
    effort: int (1-5)        — qualidade
    diarize: bool            — identificar quem fala
    denoise: bool            — remover ruído (FFmpeg)
    ultra: bool              — tudo no máximo
    hf_token: str            — HuggingFace token (para diarize)
    normalize_volume: bool   — normalizar volume (default: true)

Effort presets:
    1 — Rapido:      int8,    batch=16, beam=1
    2 — Normal:      int8,    batch=16, beam=3
    3 — Qualidade:   float16, batch=8,  beam=5
    4 — Premium:     float16, batch=4,  beam=5,  best_of=5
    5 — Ultra:       float32, batch=1,  beam=10, best_of=5, patience=2
"""
import base64
import gc
import os
import subprocess
import tempfile
import time
from collections import Counter

import runpod
import torch
import whisperx

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = os.environ.get("WHISPERX_MODEL", "large-v3")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Ensure pyannote uses same cache as huggingface hub
os.environ["PYANNOTE_CACHE"] = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))

# --- Effort → model config ---
EFFORT_CONFIG = {
    1: {"compute_type": "int8",    "batch_size": 16, "asr_options": {"beam_size": 1, "best_of": 1}},
    2: {"compute_type": "int8",    "batch_size": 16, "asr_options": {"beam_size": 3, "best_of": 1}},
    3: {"compute_type": "float16", "batch_size": 8,  "asr_options": {"beam_size": 5, "best_of": 3}},
    4: {"compute_type": "float16", "batch_size": 4,  "asr_options": {"beam_size": 5, "best_of": 5}},
    5: {"compute_type": "float32", "batch_size": 1,  "asr_options": {"beam_size": 10, "best_of": 5, "patience": 2, "temperatures": [0.0]}},
}


def _vram_info() -> str:
    if DEVICE != "cuda":
        return "CPU"
    free = torch.cuda.mem_get_info()[0] // 1024**2
    total = torch.cuda.mem_get_info()[1] // 1024**2
    return f"{free}/{total}MB"


# --- Pre-load at startup ---
print(f"WhisperX RunPod — {MODEL_NAME} on {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} — {_vram_info()}")

# Cache de modelos por compute_type
_whisper_models: dict[str, object] = {}


def _get_whisper_model(effort: int):
    config = EFFORT_CONFIG.get(effort, EFFORT_CONFIG[3])
    ct = config["compute_type"]
    if ct not in _whisper_models:
        print(f"Loading Whisper {MODEL_NAME} ({ct})... ({_vram_info()})")
        _whisper_models[ct] = whisperx.load_model(
            MODEL_NAME, DEVICE, compute_type=ct,
            asr_options=config["asr_options"],
        )
        print(f"Loaded! ({_vram_info()})")
    return _whisper_models[ct], config["batch_size"]


# Pre-load default model
_get_whisper_model(2)

# Pre-load diarize model at startup (not inside handler — avoids RunPod handler context issues)
_diarize_model = None
if HF_TOKEN:
    try:
        print(f"Pre-loading diarization model... ({_vram_info()})")
        # Step 1: pre-download all files
        from huggingface_hub import snapshot_download
        snapshot_download("pyannote/speaker-diarization-community-1", token=HF_TOKEN)
        print(f"Model files downloaded. Initializing pipeline...")
        # Step 2: init pipeline
        from whisperx.diarize import DiarizationPipeline
        _diarize_model = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
        print(f"Diarization loaded! ({_vram_info()})")
    except Exception as e:
        print(f"Diarization pre-load failed: {e}")
        _diarize_model = None
else:
    print("No HF_TOKEN — diarization disabled")

# Block ALL network calls to HuggingFace inside handler (prevents asyncio event loop blocking)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print(f"Ready! ({_vram_info()})")


def _get_diarize_model(hf_token: str = ""):
    """Returns pre-loaded diarize model or None."""
    return _diarize_model


def _denoise(audio_path: str) -> str:
    """Denoise + volume normalization via FFmpeg."""
    out = audio_path + ".clean.wav"
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-af", (
                "highpass=f=80,"
                "lowpass=f=8000,"
                "afftdn=nf=-20:nr=10:nt=w,"
                "dynaudnorm=f=150:g=15"
            ),
            "-ar", "16000", "-ac", "1",
            out,
        ], capture_output=True, timeout=120)
        if result.returncode == 0 and os.path.exists(out):
            print("Denoise OK")
            return out
    except Exception as e:
        print(f"Denoise failed: {e}")
    return audio_path


def _normalize_volume(audio_path: str) -> str:
    """Normalize volume via FFmpeg loudnorm."""
    out = audio_path + ".norm.wav"
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar", "16000", "-ac", "1",
            out,
        ], capture_output=True, timeout=60)
        if result.returncode == 0 and os.path.exists(out):
            return out
    except Exception:
        pass
    return audio_path


def _filter_hallucinations(segments: list) -> list:
    """Remove repetitive hallucinated segments."""
    filtered = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        words = text.lower().split()
        if len(words) > 5:
            most_common = Counter(words).most_common(1)[0][1]
            if most_common / len(words) > 0.6:
                continue
        filtered.append(seg)
    return filtered


def _to_native(val):
    """Convert numpy/torch types to native Python types."""
    import numpy as np
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {k: _to_native(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_native(v) for v in val]
    return val


def _serialize_segments(segments: list) -> list:
    """Convert all segment values to JSON-serializable Python types."""
    clean = []
    for seg in segments:
        s = {}
        for k, v in seg.items():
            if k == "words" and isinstance(v, list):
                s[k] = [_to_native(w) for w in v]
            else:
                s[k] = _to_native(v)
        clean.append(s)
    return clean


def handler(job):
    start_time = time.time()
    inp = job["input"]
    cleanup = []

    audio_b64 = inp.get("audio_base_64") or inp.get("audio_base64")
    if not audio_b64:
        return {"error": "audio_base_64 is required"}

    language = inp.get("language", "pt")
    effort = max(1, min(5, int(inp.get("effort", 3))))
    diarize = bool(inp.get("diarize", False))
    denoise = bool(inp.get("denoise", False))
    ultra = bool(inp.get("ultra", False))
    normalize = bool(inp.get("normalize_volume", True))
    hf_token = inp.get("hf_token", "")

    if ultra:
        effort = 5
        diarize = True
        denoise = True

    print(f"Job: effort={effort}, diarize={diarize}, denoise={denoise}, ultra={ultra}")

    # Save audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(base64.b64decode(audio_b64))
        tmp.flush()
        audio_path = tmp.name
        cleanup.append(audio_path)

    try:
        # Pre-process
        if denoise:
            runpod.serverless.progress_update(job, {"step": "Removendo ruido...", "progress": 5})
            new_path = _denoise(audio_path)
            if new_path != audio_path:
                cleanup.append(new_path)
                audio_path = new_path
        elif normalize:
            runpod.serverless.progress_update(job, {"step": "Normalizando volume...", "progress": 5})
            new_path = _normalize_volume(audio_path)
            if new_path != audio_path:
                cleanup.append(new_path)
                audio_path = new_path

        # Transcribe
        runpod.serverless.progress_update(job, {"step": f"Transcrevendo (effort {effort})...", "progress": 20})
        audio = whisperx.load_audio(audio_path)
        model, batch_size = _get_whisper_model(effort)
        result = model.transcribe(audio, batch_size=batch_size, language=language)

        # Align
        runpod.serverless.progress_update(job, {"step": "Alinhando timestamps...", "progress": 55})
        align_model, metadata = whisperx.load_align_model(language_code=language, device=DEVICE)
        result = whisperx.align(result["segments"], align_model, metadata, audio, DEVICE)
        del align_model, metadata
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Diarize
        diarize_success = False
        if diarize:
            runpod.serverless.progress_update(job, {"step": "Identificando falantes...", "progress": 70})
            try:
                dm = _get_diarize_model(hf_token)
                if dm is not None:
                    diarize_segments = dm(audio_path)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    diarize_success = True
                    print("Diarize OK")
                else:
                    print("Diarize skipped (model not available)")
            except Exception as e:
                print(f"Diarize failed: {e}")

        # Post-process
        runpod.serverless.progress_update(job, {"step": "Finalizando...", "progress": 90})
        segments = _filter_hallucinations(result.get("segments", []))
        full_text = " ".join(s.get("text", "").strip() for s in segments if s.get("text", "").strip())
        duration = audio.shape[0] / 16000

        # Serialize segments to pure Python types (no numpy/torch)
        clean_segments = _serialize_segments(segments)

        elapsed = time.time() - start_time
        print(f"Done: {len(full_text)} chars, {len(clean_segments)} segments, {elapsed:.1f}s ({_vram_info()})")

        return {
            "text": full_text,
            "segments": clean_segments,
            "language": language,
            "duration_seconds": round(float(duration), 1),
            "processing_seconds": round(elapsed, 1),
            "effort": int(effort),
            "diarize": bool(diarize),
            "denoise": bool(denoise),
            "ultra": bool(ultra),
        }

    finally:
        for f in cleanup:
            try:
                os.unlink(f)
            except Exception:
                pass
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


runpod.serverless.start({"handler": handler})

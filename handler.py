"""
RunPod Serverless Handler — WhisperX Transcription (16GB+ VRAM)

Modelos pré-carregados na GPU. Sem limitações de VRAM.
DeepFilterNet em GPU para denoise de alta qualidade.

Input:
    audio_base_64: str       — áudio em base64
    language: str            — idioma (default: "pt")
    effort: int (1-5)        — qualidade (controla compute_type, beam, batch)
    diarize: bool            — identificar quem fala
    denoise: bool            — remover ruído com DeepFilterNet (GPU)
    ultra: bool              — tudo no máximo (float32 + beam=10 + denoise GPU + diarize)
    hf_token: str            — HuggingFace token (para diarize)
    normalize_volume: bool   — normalizar volume com FFmpeg (default: true)

Output:
    text: str                — transcrição completa
    segments: list           — segmentos com timestamps, speakers, words
    language: str
    duration_seconds: float
    processing_seconds: float
    effort: int
    diarize: bool
    denoise: bool

Effort presets:
    1 — Rapido:      int8,    batch=16, beam=1
    2 — Normal:      int8,    batch=16, beam=3
    3 — Qualidade:   float16, batch=8,  beam=5
    4 — Premium:     float16, batch=4,  beam=5,  best_of=5
    5 — Ultra:       float32, batch=1,  beam=10, best_of=5, patience=2, temp=0.0
"""
import base64
import gc
import os
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

import runpod
import torch
import whisperx

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = os.environ.get("WHISPERX_MODEL", "large-v3")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# --- Effort → model config ---
EFFORT_CONFIG = {
    1: {"compute_type": "int8",    "batch_size": 16, "asr_options": {"beam_size": 1, "best_of": 1}},
    2: {"compute_type": "int8",    "batch_size": 16, "asr_options": {"beam_size": 3, "best_of": 1}},
    3: {"compute_type": "float16", "batch_size": 8,  "asr_options": {"beam_size": 5, "best_of": 3}},
    4: {"compute_type": "float16", "batch_size": 4,  "asr_options": {"beam_size": 5, "best_of": 5}},
    5: {"compute_type": "float32", "batch_size": 1,  "asr_options": {"beam_size": 10, "best_of": 5, "patience": 2, "temperatures": [0.0]}},
}

# --- Pre-load models at startup ---
def _vram_info() -> str:
    if DEVICE != "cuda":
        return "CPU mode"
    free = torch.cuda.mem_get_info()[0] // 1024**2
    total = torch.cuda.mem_get_info()[1] // 1024**2
    return f"{free}/{total}MB"


print(f"WhisperX RunPod — {MODEL_NAME} on {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} — {_vram_info()}")

# Cache de modelos whisper por compute_type (evita recarregar)
_whisper_models: dict[str, object] = {}


def _get_whisper_model(effort: int):
    """Retorna modelo Whisper para o effort level. Cacheia por compute_type."""
    config = EFFORT_CONFIG.get(effort, EFFORT_CONFIG[3])
    ct = config["compute_type"]

    if ct not in _whisper_models:
        print(f"Loading Whisper {MODEL_NAME} ({ct})... ({_vram_info()})")
        _whisper_models[ct] = whisperx.load_model(
            MODEL_NAME, DEVICE, compute_type=ct,
            asr_options=config["asr_options"],
        )
        print(f"Whisper {ct} loaded! ({_vram_info()})")

    return _whisper_models[ct], config["batch_size"]


# Pre-load default model (int8 — mais leve, pronto para requests rápidos)
_get_whisper_model(2)

# Diarize model — loaded on first use
_diarize_model = None

# DeepFilterNet — loaded on first use
_df_model = None

print(f"Ready! ({_vram_info()})")


def _get_diarize_model(hf_token: str = ""):
    global _diarize_model
    if _diarize_model is not None:
        return _diarize_model

    token = hf_token or HF_TOKEN
    if not token:
        raise ValueError("HF_TOKEN required for diarization")

    from whisperx.diarize import DiarizationPipeline
    print(f"Loading diarization model... ({_vram_info()})")
    _diarize_model = DiarizationPipeline(token=token, device=DEVICE)
    print(f"Diarization loaded! ({_vram_info()})")
    return _diarize_model


def _denoise_gpu(audio_path: str) -> str:
    """DeepFilterNet em GPU — remoção de ruído de alta qualidade."""
    global _df_model
    try:
        if _df_model is None:
            from df.enhance import init_df
            print(f"Loading DeepFilterNet (GPU)... ({_vram_info()})")
            _df_model = init_df(config_allow_defaults=True)
            print(f"DeepFilterNet loaded! ({_vram_info()})")

        from df.enhance import enhance, load_audio, save_audio
        model, df_state, _ = _df_model
        audio, sr = load_audio(audio_path, sr=df_state.sr())
        enhanced = enhance(model, df_state, audio)
        out = audio_path + ".clean.wav"
        save_audio(out, enhanced, sr)
        print("Denoise OK (GPU)")
        return out
    except Exception as e:
        print(f"DeepFilterNet failed, falling back to FFmpeg: {e}")
        return _denoise_ffmpeg(audio_path)


def _denoise_ffmpeg(audio_path: str) -> str:
    """Fallback: denoise via FFmpeg (CPU)."""
    out = audio_path + ".ffmpeg.wav"
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-af", "highpass=f=80,lowpass=f=8000,afftdn=nf=-20:nr=10:nt=w,dynaudnorm=f=150:g=15",
            "-ar", "16000", "-ac", "1",
            out,
        ], capture_output=True, timeout=120)
        if result.returncode == 0 and os.path.exists(out):
            return out
    except Exception as e:
        print(f"FFmpeg denoise failed: {e}")
    return audio_path


def _normalize_volume(audio_path: str) -> str:
    """Normaliza volume com FFmpeg loudnorm."""
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
    """Remove segmentos com texto repetitivo (hallucinations do Whisper)."""
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


def _cleanup_files(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass


def handler(job):
    """RunPod handler — transcrição completa com todas as opções."""
    start_time = time.time()
    job_input = job["input"]
    cleanup = []

    # Parse input
    audio_b64 = job_input.get("audio_base_64") or job_input.get("audio_base64")
    if not audio_b64:
        return {"error": "audio_base_64 is required"}

    language = job_input.get("language", "pt")
    effort = max(1, min(5, int(job_input.get("effort", 3))))
    diarize = bool(job_input.get("diarize", False))
    denoise = bool(job_input.get("denoise", False))
    ultra = bool(job_input.get("ultra", False))
    normalize = bool(job_input.get("normalize_volume", True))
    hf_token = job_input.get("hf_token", "")

    # Ultra = tudo no máximo
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
        # Step 1: Pre-processing
        if denoise:
            runpod.serverless.progress_update(job, {"step": "Removendo ruido (GPU)...", "progress": 5})
            new_path = _denoise_gpu(audio_path)
            if new_path != audio_path:
                cleanup.append(new_path)
                audio_path = new_path

        if normalize:
            runpod.serverless.progress_update(job, {"step": "Normalizando volume...", "progress": 10})
            new_path = _normalize_volume(audio_path)
            if new_path != audio_path:
                cleanup.append(new_path)
                audio_path = new_path

        # Step 2: Transcribe
        runpod.serverless.progress_update(job, {"step": f"Transcrevendo (effort {effort})...", "progress": 20})
        audio = whisperx.load_audio(audio_path)
        model, batch_size = _get_whisper_model(effort)
        result = model.transcribe(audio, batch_size=batch_size, language=language)

        # Step 3: Align (word-level timestamps)
        runpod.serverless.progress_update(job, {"step": "Alinhando timestamps...", "progress": 55})
        align_model, metadata = whisperx.load_align_model(language_code=language, device=DEVICE)
        result = whisperx.align(result["segments"], align_model, metadata, audio, DEVICE)
        del align_model, metadata
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Step 4: Diarize
        if diarize:
            runpod.serverless.progress_update(job, {"step": "Identificando falantes...", "progress": 70})
            try:
                dm = _get_diarize_model(hf_token)
                diarize_segments = dm(audio_path)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                print(f"Diarize failed: {e}")

        # Step 5: Post-process
        runpod.serverless.progress_update(job, {"step": "Finalizando...", "progress": 90})
        segments = _filter_hallucinations(result.get("segments", []))
        full_text = " ".join(s.get("text", "").strip() for s in segments if s.get("text", "").strip())
        duration = audio.shape[0] / 16000

        elapsed = time.time() - start_time
        print(f"Done: {len(full_text)} chars, {len(segments)} segments, {elapsed:.1f}s ({_vram_info()})")

        return {
            "text": full_text,
            "segments": segments,
            "language": language,
            "duration_seconds": round(duration, 1),
            "processing_seconds": round(elapsed, 1),
            "effort": effort,
            "diarize": diarize,
            "denoise": denoise,
            "ultra": ultra,
        }

    finally:
        _cleanup_files(*cleanup)
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


runpod.serverless.start({"handler": handler})

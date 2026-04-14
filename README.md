# WhisperX Worker — RunPod Serverless

Speech-to-Text serverless worker powered by WhisperX, running on RunPod GPUs.

## Features

- **WhisperX large-v3** — state-of-the-art transcription
- **5 quality levels** — from fast (int8, beam=1) to ultra (float32, beam=10)
- **Speaker diarization** — identifies who is speaking (pyannote)
- **GPU denoising** — DeepFilterNet removes background noise
- **Volume normalization** — FFmpeg loudnorm
- **Hallucination filtering** — removes repetitive Whisper artifacts
- **Word-level timestamps** — precise alignment

## Quality Levels

| Effort | Compute | Beam | Diarize | Denoise | Speed |
|--------|---------|------|---------|---------|-------|
| 1 Rapido | int8 | 1 | - | - | ~10x |
| 2 Normal | int8 | 3 | - | - | ~7x |
| 3 Qualidade | float16 | 5 | - | - | ~4x |
| 4 Premium | float16 | 5 | Yes | - | ~2x |
| 5 Ultra | float32 | 10 | Yes | Yes | ~1x |

## API

### Input

```json
{
  "input": {
    "audio_base_64": "BASE64_ENCODED_AUDIO",
    "language": "pt",
    "effort": 3,
    "diarize": false,
    "denoise": false,
    "ultra": false,
    "normalize_volume": true,
    "hf_token": "hf_xxx"
  }
}
```

### Output

```json
{
  "text": "Full transcription text...",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello world",
      "speaker": "SPEAKER_00",
      "words": [...]
    }
  ],
  "language": "pt",
  "duration_seconds": 120.0,
  "processing_seconds": 15.3,
  "effort": 3,
  "diarize": false,
  "denoise": false,
  "ultra": false
}
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `WHISPERX_MODEL` | No | `large-v3` | Whisper model name |
| `HF_TOKEN` | For diarize | - | HuggingFace token (pyannote) |

## Deploy on RunPod

1. Connect GitHub to RunPod (Settings > GitHub > Connect)
2. New Endpoint > Deploy from GitHub > Select this repo
3. Set GPU: 16GB+ (A4000, 4090, etc)
4. Set env vars: `HF_TOKEN`
5. Deploy

## License

MIT

# FasterWhisperModel — Realtime microphone transcription ✅

This repository contains simple examples to run Faster Whisper for file and realtime microphone transcription on Windows.

## Files

- `fasterWhisper.py` — Example of transcribing an audio file with `WhisperModel.transcribe()`.
- `realtime.py` — Realtime microphone capture and streaming transcription (improved).
- `requirements.txt` — Python requirements (see notes for Torch/CUDA).

---

## Requirements 🔧

- Python 3.8+
- Install Python packages:

```bash
pip install -r requirements.txt
```

- Torch: On Windows, install the correct `torch` build for your CUDA version (or CPU-only). See https://pytorch.org/get-started/locally/ for the recommended pip command.
- ffmpeg: Required for some audio file operations. Install from https://ffmpeg.org/download.html if needed.

---

## Realtime usage 💡

Run the realtime script with defaults (medium.en model, CUDA if available):

```bash
python realtime.py
```

Options:

- `--model` Model name (default `medium.en`)
- `--device` `cuda` or `cpu` (default `cuda`)
- `--samplerate` Target sampling rate (default `16000`)
- `--block-duration` Microphone block duration in seconds (default `0.5`)
- `--chunk-duration` Transcription chunk duration in seconds (default `2.0`)
- `--channels` Number of input channels (default `1`)
- `--beam` Beam size for transcription (1 = fastest)

Example (CPU):

```bash
python realtime.py --device cuda --model medium.en
```

Notes:
- The script now preserves remainder audio between chunks to avoid dropping audio.
- The audio queue has a max size to avoid unbounded memory growth; older audio will be dropped when the host cannot keep up.
- If your microphone does not support the requested samplerate, you may get an error — try a compatible samplerate or adjust `--samplerate`.

---

## File transcription example

`fasterWhisper.py` shows how to transcribe a file; remember to install `ffmpeg` if you rely on ffmpeg-backed loaders.

---

## Troubleshooting ⚠️

- Microphone errors: Check that your default input device is set and not in use by other apps.
- If `sounddevice` installation fails on Windows, ensure you have the appropriate wheels or install Visual C++ build tools.
- For GPU use: install the correct `torch` wheel for your CUDA version.
- If you see an error mentioning `cublas64_12.dll` (or similar), it means the CUDA runtime required by CTranslate2 / Faster Whisper isn't available on your system. On Windows:
  1. Install the matching CUDA Toolkit (e.g., CUDA 12.x) from NVIDIA: https://developer.nvidia.com/cuda-downloads
  2. Ensure the CUDA `bin` directory is on your PATH (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin`).
  3. Update your NVIDIA drivers.
  4. If using conda: `conda install -c conda-forge cudatoolkit=12.1` can provide the runtime in the environment.
  You can also run the script with `--device cpu` as a quick workaround.

- Quick diagnostic: run `python realtime.py --diag --device cuda` to check for common issues (checks for `cublas` DLLs and `nvidia-smi`). The command will print findings and a short recommendation and then exit with code 0 if CUDA looks available or non-zero if it is not.

If you want further enhancements (VAD, streaming partial results, or websocket integration), tell me which direction you'd like to go next. 

import sounddevice as sd   # audio from microphone
import numpy as np         # to handle audio as numbers
import queue               # to safely store audio blocks 
import threading           # to run mic and transcription parallely
from faster_whisper import WhisperModel

import argparse
import logging
import sys

# Settings (can be overridden via CLI)
parser = argparse.ArgumentParser(description="Realtime microphone transcription with Faster Whisper")
parser.add_argument("--model", default="medium.en", help="Model name (e.g., medium.en, large-v3)")
parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
parser.add_argument("--samplerate", type=int, default=16000, help="Target sampling rate")
parser.add_argument("--block-duration", type=float, default=0.5, help="Microphone block duration (seconds)")
parser.add_argument("--chunk-duration", type=float, default=2.0, help="Transcription chunk duration (seconds)")
parser.add_argument("--channels", type=int, default=1, help="Number of input channels")
parser.add_argument("--beam", type=int, default=1, help="Beam size for transcription (1 = fastest)")
parser.add_argument("--diag", action="store_true", help="Run CUDA diagnostic checks (checks cublas DLL and nvidia-smi) and exit")
args = parser.parse_args()

samplerate = args.samplerate   # audio samples per second
block_duration = args.block_duration #seconds. Given by mic
chunk_duration = args.chunk_duration   #seconds. Shared with whisper
channels = args.channels

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

# Use a bounded queue to avoid unlimited memory growth
audio_queue = queue.Queue(maxsize=100)
audio_buffer = []

# Model setup
# Normalize device aliases
device = args.device.lower()
if device == "gpu":
  device = "cuda"
if device not in ("cuda", "cpu"):
  print(f"Unknown device '{args.device}', defaulting to 'cuda'")
  device = "cuda"


def check_cuda_runtime():
  """Check for common CUDA runtime DLLs and nvidia-smi. Return True if a reasonable runtime is detected."""
  import ctypes, subprocess, os

  candidates = ["cublas64_12.dll", "cublas64_11.dll"]
  loaded = []
  for dll in candidates:
    try:
      ctypes.cdll.LoadLibrary(dll)
      loaded.append(dll)
    except OSError:
      pass
  if loaded:
    print(f"CUDA runtime loaded in process: {loaded}")
    return True

  # Search PATH for DLL files
  path_dirs = os.environ.get("PATH", "").split(os.pathsep)
  found_paths = []
  for d in path_dirs:
    try:
      for dll in candidates:
        p = os.path.join(d, dll)
        if os.path.isfile(p):
          found_paths.append(p)
    except Exception:
      pass
  if found_paths:
    print("Found CUDA DLL(s) on PATH:")
    for p in found_paths:
      print("  " + p)
    return True

  # Try nvidia-smi for driver/GPU detection
  try:
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
    print("nvidia-smi output:")
    print(out.stdout.strip())
  except Exception:
    print("nvidia-smi not found or failed to run.")

  # Nothing found
  return False

# If user requested diagnostics, run check and exit
if args.diag:
  print("Running CUDA diagnostics (--diag)...")
  ok = check_cuda_runtime()
  if ok:
    print("CUDA runtime appears available.")
    sys.exit(0)
  else:
    print("CUDA runtime not detected. See README Troubleshooting for steps to install CUDA or use conda 'cudatoolkit'.")
    sys.exit(1)

try:
  # Pre-check for CUDA runtime when using GPU to provide clearer errors earlier
  if device == "cuda" and not check_cuda_runtime():
    print("CUDA runtime not detected before loading the model. Aborting. See README for troubleshooting.")
    sys.exit(1)

  compute_type = "float16" if device == "cuda" else "int8"
  model = WhisperModel(args.model, device=device, compute_type=compute_type)
except Exception as e:
  err = str(e)
  # Helpful guidance for common CUDA DLL errors on Windows
  if "cublas" in err.lower() or "cublas64" in err.lower():
    print("Failed to initialize GPU backend: missing or incompatible CUDA runtime (cublas DLL).")
    print("Common fixes on Windows:")
    print("  1) Install the CUDA Toolkit matching your library (e.g., CUDA 12.x) from https://developer.nvidia.com/cuda-downloads")
    print("  2) Make sure the CUDA 'bin' folder is on your PATH (e.g., C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin)")
    print("  3) Update your NVIDIA GPU driver to the latest version")
    print("  4) If you use conda, you can alternatively install 'cudatoolkit' in your env: `conda install -c conda-forge cudatoolkit=12.1`")
    print("As a workaround, run with --device cpu to use the CPU instead.")
    print(f"Original error: {err}")
    sys.exit(1)
  else:
    print(f"Failed to load model {args.model} on device {args.device}: {err}")
    raise

# Activates when mic gives audio
def audio_callback(indata, frames, time, status):
  if status:
    print(status)
  # Try not to block the audio callback. Drop oldest blocks when full.
  try:
    audio_queue.put_nowait(indata.copy())
  except queue.Full:
    try:
      _ = audio_queue.get_nowait()  # drop one
      audio_queue.put_nowait(indata.copy())
    except Exception:
      pass

# Keeps the mic on
def recorder():
  try:
    with sd.InputStream(samplerate=samplerate, channels=channels, dtype='float32', callback=audio_callback, blocksize=frames_per_block):
      print(f"Listening... Press Ctrl+C to stop. (samplerate={samplerate}, chunk={chunk_duration}s)")
      while True:
        sd.sleep(1000)
  except Exception as e:
    print(f"Input stream error: {e}")
    raise

# Transcriber: collects blocks, forms fixed-size chunks and keeps remainders
def transcriber():
  global audio_buffer
  while True:
    block = audio_queue.get()
    audio_buffer.append(block)

    # Transcription starts when we have enough frames
    total_frames = sum(b.shape[0] for b in audio_buffer)
    if total_frames >= frames_per_chunk:
      data = np.concatenate(audio_buffer, axis=0)
      chunk = data[:frames_per_chunk]
      remainder = data[frames_per_chunk:]
      audio_buffer = [remainder] if remainder.size else []

      # If multiple channels, mixdown to mono
      if chunk.ndim > 1 and chunk.shape[1] > 1:
        chunk = np.mean(chunk, axis=1)

      audio_data = chunk.flatten().astype(np.float32)

      try:
        segments, _ = model.transcribe(
          audio_data,
          language="en",
          beam_size=args.beam   # Max speed when low
        )

        for segment in segments:
          text = getattr(segment, 'text', None) or getattr(segment, 'txt', '')
          print(text.strip())    # Print just text
      except Exception as e:
        print(f"Transcription error: {e}")



# Start threads for real-time
if __name__ == "__main__":
    try:
        # Start recorder thread
        threading.Thread(target=recorder, daemon=True).start()
        transcriber()
    except KeyboardInterrupt:
        print("\nStopped by user. Exiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        raise

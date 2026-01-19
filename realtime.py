import sounddevice as sd   # audio from microphone
import numpy as np         # to handle audio as numbers
import queue               # to safely store audio blocks 
import threading           # to run mic and transcription parallely
from faster_whisper import WhisperModel

# Settings
samplerate = 16000   # audio samples per second
block_duration = 0.5 #seconds. Given by mic
chunk_duration = 2   #seconds. Shared with whisper
channels = 1

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []

# Model setup: medium.en + float16 (for 3060)
model = WhisperModel("medium.en", device = "cuda", compute_type = "float16")
# model = WhisperModel("medium.en", device = "cpu", compute_type = "int8")


# Activates when mic gives audio
def audio_callback(intdata, frames, time, status):
  if status:
    print(status)
  audio_queue.put(intdata.copy())

# Keeps the mic on
def recorder():
  with sd.InputStream(samplerate=samplerate, channels=channels, callback=audio_callback, blocksize=frames_per_block):
    print("Listening... Press Ctrl+C to stop.")
    while True:
      sd.sleep(100)

def transcriber():
  global audio_buffer
  while True:
    block = audio_queue.get()
    audio_buffer.append(block)

    # Transcription starts when the buffer gets 2 sec worth audio
    # Joining all the blocks and cutting irrelevant sound
    # Converting in whisper format
    total_frames = sum(len(b) for b in audio_buffer)
    if total_frames >= frames_per_chunk:
      audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
      audio_buffer = []   #Clear buffer
      audio_data = audio_data.flatten().astype(np.float32)

      # Transcription without timestamps
      segments, _ = model.transcribe(
        audio_data,
        language="en",
        beam_size=1   # Max speed. The lower, the better
      )

      for segment in segments:
        print(f"{segment.text}")    #Just print text, no timestamps



# Start threads for real-time
threading.Thread(target=recorder, daemon=True).start()
transcriber()

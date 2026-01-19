from faster_whisper import WhisperModel

# model_size = "large-v3"
model_size = "medium.en"

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")
model = WhisperModel(model_size, device="cpu", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, _ = model.transcribe("audio.mp3", language="en", beam_size=5)

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    text = getattr(segment, 'text', None) or getattr(segment, 'txt', '')
    print(text.strip())
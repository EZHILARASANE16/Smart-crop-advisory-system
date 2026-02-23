import whisper
import sounddevice as sd
import numpy as np
import tempfile
import os
import queue
import sys
import threading

# Load model once
model = whisper.load_model("small")

# Recording setup
q = queue.Queue()
samplerate = 16000  # Whisper prefers 16kHz
blocksize = 8000

def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

def listen_and_transcribe():
    with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback, blocksize=blocksize):
        print("🎤 Listening... Press Ctrl+C to stop.")
        audio_buffer = np.zeros((0, 1), dtype=np.float32)

        try:
            while True:
                while not q.empty():
                    data = q.get()
                    audio_buffer = np.append(audio_buffer, data)

                if len(audio_buffer) > samplerate * 3:  # every 3 seconds
                    # Save temporary wav
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                        wav_path = tmpfile.name

                    sd.write(wav_path, audio_buffer, samplerate)
                    
                    # Transcribe
                    result = model.transcribe(wav_path)
                    print("📝", result["text"])

                    # reset buffer
                    audio_buffer = np.zeros((0, 1), dtype=np.float32)

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")

if __name__ == "__main__":
    listen_and_transcribe()

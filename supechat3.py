import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import queue
import threading
import time
from collections import deque
import requests
from concurrent.futures import ThreadPoolExecutor

class VoiceEnabledChatbot:
    def __init__(self):
        # Audio configuration
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1

        # Load Whisper medium model (optimized for CUDA int811)
        print("Loading Whisper medium model for voice recognition...")
        self.whisper_model = WhisperModel("medium", device="cuda", compute_type="int8")
        print("Whisper model loaded successfully.")

        # Groq API configuration
        self.API_KEY = "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"
        self.API_URL = "https://api.groq.com/openai/v1/chat/completions"
        self.MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

        # Buffers & state
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.recent_transcripts = deque(maxlen=5)
        self.min_similarity_threshold = 0.7

        # Thread pool for asynchronous API calls
        self.executor = ThreadPoolExecutor(max_workers=2)

        # System prompt
        self.system_prompt = """You are an AI assistant for a Smart Crop Advisory System designed for small and marginal farmers.
Provide practical, actionable advice on:
- Crop selection based on soil type, season, and climate
- Irrigation scheduling and water management
- Fertilizer recommendations and pest control
- Market prices and best practices for small farmers
- Weather-based farming decisions
Keep responses concise and farmer-friendly."""

    def audio_callback(self, indata, frames, time_info, status):
        if self.is_listening:
            # Copy chunk into queue
            self.audio_queue.put(indata[:, 0].copy())

    def is_duplicate(self, text):
        new_words = set(text.lower().split())
        for old in self.recent_transcripts:
            old_words = set(old.lower().split())
            if old_words and new_words:
                sim = len(old_words & new_words) / len(old_words | new_words)
                if sim >= self.min_similarity_threshold:
                    return True
        return False

    def transcribe_vad(self, audio_buffer):
        # VAD-filtered transcription
        segments, info = self.whisper_model.transcribe(
            audio_buffer,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6
        )
        text = ""
        for seg in segments:
            t = seg.text.strip()
            if t and not self.is_duplicate(t):
                text += t + " "
                self.recent_transcripts.append(t)
        return text.strip()

    def fetch_and_print(self, user_text):
        # Prepare request
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text}
        ]
        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.MODEL_NAME,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.7,
            "top_p": 1.0
        }
        try:
            resp = requests.post(self.API_URL, headers=headers, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices")
            if choices:
                print(f"🤖 Assistant: {choices[0]['message']['content']}\n")
            else:
                print("🤖 Assistant: I couldn't generate a response.\n")
        except Exception as e:
            print(f"🤖 Assistant: Sorry, I'm having trouble right now. ({e})\n")

    def process_audio(self):
        buffer = np.zeros((0,), dtype=np.float32)
        while self.is_listening:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                buffer = np.concatenate((buffer, chunk))
                # When 0.5s of audio accumulated, transcribe
                if len(buffer) >= self.SAMPLE_RATE // 2:
                    segment = buffer[: self.SAMPLE_RATE // 2]
                    buffer = buffer[self.SAMPLE_RATE // 2 :]
                    if np.max(np.abs(segment)) > 0.01:
                        text = self.transcribe_vad(segment)
                        if text:
                            print(f"🗣️ You: {text}")
                            self.executor.submit(self.fetch_and_print, text)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")

    def start_voice(self):
        self.is_listening = True
        # Persistent stream reference
        self.stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            callback=self.audio_callback,
            blocksize=int(self.SAMPLE_RATE * 0.25),  # 0.25s blocks
            dtype="float32",
            latency="low"
        )
        threading.Thread(target=self.process_audio, daemon=True).start()
        self.stream.start()
        print("🎤 Voice chat active (press Ctrl+C to stop)\n")
        try:
            while self.is_listening:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.is_listening = False
            self.stream.stop()
            self.stream.close()

    def text_mode(self):
        print("💬 Text Chat Mode - Type 'quit' to exit.\n")
        while True:
            user = input("You: ").strip()
            if user.lower() in ("quit", "exit"):
                print("🤖 Assistant: Goodbye!")
                break
            print("🤖 Assistant:", end=" ", flush=True)
            self.fetch_and_print(user)

def main():
    print("="*50)
    print("🌾 SMART CROP ADVISORY SYSTEM (SIH25010)")
    print("="*50)
    bot = VoiceEnabledChatbot()
    mode = input("Choose 1) Voice Chat  2) Text Chat: ").strip()
    if mode == "1":
        bot.start_voice()
    else:
        bot.text_mode()

if __name__ == "__main__":
    main()

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import queue
import threading
import time
from collections import deque
import requests

class VoiceEnabledChatbot:
    def __init__(self):
        # Audio configuration
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1
        self.CHUNK_DURATION = 1.0           # 1 second per chunk
        self.CHUNK_SIZE = int(self.SAMPLE_RATE * self.CHUNK_DURATION)
        
        # Whisper model
        print("Loading Whisper model...")
        self.whisper_model = WhisperModel("medium", device="cuda", compute_type="int8")
        print("Model loaded.")

        # Groq API
        self.API_KEY = "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"
        self.API_URL = "https://api.groq.com/openai/v1/chat/completions"
        self.MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

        # Buffers & state
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.buffer = np.zeros(0, dtype=np.float32)
        self.recent_transcripts = deque(maxlen=5)
        self.min_similarity_threshold = 0.7

        # SIH25010 prompt
        self.system_prompt = """You are an AI assistant for small and marginal farmers.
Provide practical, actionable crop advice in simple language."""

    def audio_callback(self, indata, frames, time_info, status):
        if self.is_listening:
            self.audio_queue.put(indata[:, 0].copy())

    def is_duplicate(self, text):
        new = set(text.lower().split())
        for old in self.recent_transcripts:
            old_set = set(old.lower().split())
            sim = len(old_set & new) / len(old_set | new) if old_set and new else 0
            if sim >= self.min_similarity_threshold:
                return True
        return False

    def transcribe(self, chunk):
        segments, info = self.whisper_model.transcribe(
            chunk, beam_size=1, vad_filter=True,
            condition_on_previous_text=False, no_speech_threshold=0.6
        )
        out = ""
        for seg in segments:
            t = seg.text.strip()
            if t and not self.is_duplicate(t):
                out += t + " "
                self.recent_transcripts.append(t)
        return out.strip(), getattr(info, "language", "unknown")

    def get_response(self, usr):
        msgs = [{"role":"system","content":self.system_prompt},
                {"role":"user","content":usr}]
        hdr = {"Authorization":f"Bearer {self.API_KEY}",
               "Content-Type":"application/json"}
        pld = {"model":self.MODEL_NAME,
               "messages":msgs,
               "max_tokens":300,
               "temperature":0.7,
               "top_p":1.0}
        r = requests.post(self.API_URL, headers=hdr, json=pld, timeout=15)
        if r.status_code!=200:
            print(f"API Error {r.status_code}: {r.text}")
            return "Sorry, I'm having trouble."
        ch = r.json().get("choices")
        return ch[0]["message"]["content"] if ch else "I couldn't respond."

    def process_audio(self):
        print("🎙️ Voice mode active.")
        while self.is_listening:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                self.buffer = np.concatenate((self.buffer, chunk))
                if len(self.buffer) >= self.CHUNK_SIZE:
                    buf = self.buffer[:self.CHUNK_SIZE]
                    self.buffer = self.buffer[self.CHUNK_SIZE:]
                    if np.max(np.abs(buf)) > 0.01:
                        txt, lang = self.transcribe(buf)
                        if txt:
                            print(f"🗣️ [{lang}] {txt}")
                            resp = self.get_response(txt)
                            print(f"🤖 {resp}\n")
            except queue.Empty:
                continue

    def start_voice(self):
        self.is_listening = True
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
        print("🎤 Listening...")
        try:
            while self.is_listening:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.is_listening=False
            self.stream.stop()
            self.stream.close()

    def text_mode(self):
        print("💬 Text mode. Type 'quit' to exit.")
        while True:
            usr = input("You: ").strip()
            if usr.lower() in ("quit","exit"):
                print("Goodbye!")
                break
            print("🤖", self.get_response(usr), "\n")

def main():
    print("="*40)
    print("SIH25010 Smart Crop Advisory")
    print("="*40)
    bot=VoiceEnabledChatbot()
    print("1. Voice  2. Text")
    c=input("Select: ").strip()
    bot.start_voice() if c=="1" else bot.text_mode()

if __name__=="__main__":
    main()

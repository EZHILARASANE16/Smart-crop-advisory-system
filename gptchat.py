"""
gptchat.py
Voice <-> Groq Chatbot integration.

Requirements:
    pip install faster-whisper sounddevice numpy pyttsx3 requests

Notes:
- Set your Groq API key in the environment variable GROQ_API_KEY before running.
  PowerShell (temporary for session): $env:GROQ_API_KEY="gsk_..."
  Or set it permanently in Windows Environment Variables.

- Default Whisper model is "medium" (change to "small" if low resources,
  or "large-v3" if you have a beefy GPU and want best accuracy).

- This script avoids printing the API key anywhere.
"""

import os
import queue
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import pyttsx3
import requests
from faster_whisper import WhisperModel

# ---------------------------
# Config (tweak these)
# ---------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 3.0                # seconds per chunk (latency tradeoff)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
OVERLAP_SECONDS = 0.6               # overlap to avoid cutting words
OVERLAP_SIZE = int(SAMPLE_RATE * OVERLAP_SECONDS)

MIN_AUDIO_ENERGY = 0.007            # silence threshold (adjust if needed)
RECENT_TRANSCRIPTS_MAX = 16
JACCARD_DUPLICATE_THRESHOLD = 0.78

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")  # small | medium | large-v3
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
MAX_TOKENS = 512
TEMPERATURE = 0.2

# ---------------------------
# Environment / API key
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "Please set the GROQ_API_KEY environment variable before running.\n"
        "PowerShell (temporary): $env:GROQ_API_KEY='gsk_...'\n"
        "Do NOT hard-code the key in the script."
    )

# ---------------------------
# Utilities
# ---------------------------
def jaccard_similarity(a: str, b: str) -> float:
    A = set(a.lower().split())
    B = set(b.lower().split())
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

# ---------------------------
# LiveTranscriber
# ---------------------------
class LiveTranscriber:
    def __init__(self, model_name=WHISPER_MODEL_NAME, device="cuda"):
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.chunk_size = CHUNK_SIZE
        self.overlap_size = OVERLAP_SIZE

        self.recent_transcripts = deque(maxlen=RECENT_TRANSCRIPTS_MAX)
        self.dup_threshold = JACCARD_DUPLICATE_THRESHOLD

        # Load Whisper model (compute_type chosen for device)
        compute_type = "float16" if device.startswith("cuda") else "int8_float16"
        print(f"[Transcriber] Loading Whisper model '{model_name}' on device={device} (compute_type={compute_type}) ...")
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        print("[Transcriber] Model loaded.")

        self.audio_queue = queue.Queue()
        self.buffer = np.zeros(0, dtype=np.float32)
        self.is_running = False

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            # show status but continue
            print("[Audio] status:", status)
        # mono: take first channel
        samples = indata[:, 0].astype(np.float32)
        self.audio_queue.put(samples)

    def is_duplicate(self, new_text):
        for old in self.recent_transcripts:
            if jaccard_similarity(new_text, old) >= self.dup_threshold:
                return True
        return False

    def process_loop(self, on_transcript_callback):
        """Consume audio chunks from queue, transcribe and call the callback."""
        print("[Transcriber] processing loop started.")
        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=0.12)
                self.buffer = np.concatenate([self.buffer, chunk])

                if len(self.buffer) < self.chunk_size:
                    continue

                audio_chunk = self.buffer[:self.chunk_size]
                # keep overlap in buffer
                self.buffer = self.buffer[self.chunk_size - self.overlap_size :]

                # simple energy-based VAD (avoid sending silence)
                peak = float(np.max(np.abs(audio_chunk)))
                if peak < MIN_AUDIO_ENERGY:
                    continue

                # Transcribe (returns segments, info)
                try:
                    segments, info = self.model.transcribe(
                        audio_chunk,
                        beam_size=3,
                        vad_filter=True,
                        language=None,
                        condition_on_previous_text=False,
                        word_timestamps=False,
                    )
                except Exception as e:
                    print("[Transcriber] model.transcribe error:", e)
                    continue

                text = " ".join([seg.text.strip() for seg in segments]).strip()
                if not text:
                    continue

                if self.is_duplicate(text):
                    # suppress near-duplicates
                    # print("[Transcriber] duplicate suppressed")
                    continue

                self.recent_transcripts.append(text)
                on_transcript_callback(text, getattr(info, "language", "unknown"))

            except queue.Empty:
                continue
            except Exception as e:
                print("[Transcriber] unexpected error in loop:", e)
                # small sleep to avoid tight error loop
                time.sleep(0.2)
                continue
        print("[Transcriber] processing loop stopped.")

    def start(self, on_transcript_callback):
        self.is_running = True
        t = threading.Thread(target=self.process_loop, args=(on_transcript_callback,), daemon=True)
        t.start()
        try:
            with sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                blocksize=2048,
                dtype="float32",
            ):
                print("[Audio] Listening... Press Ctrl+C to stop.")
                while self.is_running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[Audio] KeyboardInterrupt -> stopping.")
        finally:
            self.is_running = False

    def stop(self):
        self.is_running = False

# ---------------------------
# Groq chat (requests-based)
# ---------------------------
def chat_with_groq(system_prompt: str, user_message: str, history=None) -> str:
    """
    Sends a chat completion request to Groq and returns assistant text.
    history: optional list of {"role": "user"/"assistant", "content": "..."}
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # build messages: system -> history -> user
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # safe navigation: different APIs may structure slightly differently
        if "choices" in data and len(data["choices"]) > 0:
            # Try standard OpenAI-like shape
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"].strip()
            elif "text" in choice:
                return choice["text"].strip()
        # fallback: try top-level content
        return str(data)
    except requests.exceptions.RequestException as e:
        print("[Groq] Request error:", e)
        return "Sorry, I couldn't reach the chat service."
    except Exception as e:
        print("[Groq] Unexpected error parsing response:", e)
        return "Sorry, the chat service returned an unexpected response."

# ---------------------------
# TTS (pyttsx3)
# ---------------------------
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 160)
tts_engine.setProperty("volume", 0.95)

def speak_text(text: str):
    def _speak():
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print("[TTS] error:", e)
    threading.Thread(target=_speak, daemon=True).start()

# ---------------------------
# Voice + Chat bot
# ---------------------------
class VoiceGroqBot:
    def __init__(self, system_prompt=None, device="cuda", keep_history=True, history_size=6):
        self.system_prompt = system_prompt or "You are a helpful assistant for a student project. Keep responses friendly and concise."
        # small message history so the model has context
        self.keep_history = keep_history
        self.history = deque(maxlen=history_size)
        self.transcriber = LiveTranscriber(model_name=WHISPER_MODEL_NAME, device=device)

    def on_transcript(self, text: str, language: str):
        # Got a fresh transcription
        print(f"\n[User ({language})]: {text}")

        # Build history for the request
        history_list = list(self.history) if self.keep_history else None

        reply = chat_with_groq(self.system_prompt, text, history=history_list)
        print("[Assistant]:", reply)

        # speak it
        speak_text(reply)

        # append to history
        if self.keep_history:
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply})

    def start(self):
        # auto-detect device (cuda if torch available)
        device = "cuda"
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"

        # If the loaded transcriber isn't on the desired device, we won't re-create here.
        # For most setups the transcriber was loaded with the selected device already.
        print("[Bot] Starting (transcriber device assumed).")
        try:
            self.transcriber.start(self.on_transcript)
        except Exception as e:
            print("[Bot] error while running:", e)
        finally:
            self.transcriber.stop()

    def stop(self):
        self.transcriber.stop()

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    # Quick user tips if script is used directly
    print("gptchat.py — Voice -> Groq Chatbot")
    print("Make sure GROQ_API_KEY is set and you installed required packages.")
    print(f"Using Whisper model: {WHISPER_MODEL_NAME}")
    # Start bot
    # Choose device automatically
    try:
        import torch
        device_choice = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device_choice = "cpu"

    bot = VoiceGroqBot(device=device_choice)
    try:
        bot.start()
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user — exiting.")
        bot.stop()

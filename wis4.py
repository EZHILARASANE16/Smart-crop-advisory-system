import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import queue
import threading
import time
from collections import deque

class LiveTranscriber:
    def __init__(self):
        # Audio configuration: 16 kHz mono
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1
        self.CHUNK_DURATION = 5.0  # seconds
        self.CHUNK_SIZE = int(self.SAMPLE_RATE * self.CHUNK_DURATION)

        # Load large-v3 model for accuracy
        print("Loading Whisper large-v3 model... (~3 GB download first run)")
        self.model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        print("Model loaded successfully!")

        self.audio_queue = queue.Queue()
        self.is_running = False
        self.buffer = np.array([], dtype=np.float32)
        
        # Duplicate prevention
        self.recent_transcripts = deque(maxlen=10)  # Store last 10 transcripts
        self.min_similarity_threshold = 0.8  # 80% similarity = duplicate

    def audio_callback(self, indata, frames, time, status):
        audio_data = indata[:, 0].astype(np.float32)
        self.audio_queue.put(audio_data)

    def is_duplicate(self, new_text):
        """Check if transcription is duplicate of recent ones"""
        new_words = set(new_text.lower().split())
        
        for old_text in self.recent_transcripts:
            old_words = set(old_text.lower().split())
            if len(new_words) == 0 or len(old_words) == 0:
                continue
                
            # Calculate Jaccard similarity (intersection over union)
            intersection = len(new_words.intersection(old_words))
            union = len(new_words.union(old_words))
            similarity = intersection / union if union > 0 else 0
            
            if similarity >= self.min_similarity_threshold:
                return True
        return False

    def process_audio(self):
        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                self.buffer = np.concatenate([self.buffer, chunk])

                if len(self.buffer) >= self.CHUNK_SIZE:
                    audio_chunk = self.buffer[:self.CHUNK_SIZE]
                    # Reduce overlap to minimize duplicates
                    self.buffer = self.buffer[self.CHUNK_SIZE // 2:]  # 50% overlap instead of 1/3

                    if np.max(np.abs(audio_chunk)) > 0.01:
                        try:
                            segments, info = self.model.transcribe(
                                audio_chunk,
                                beam_size=1,  # Reduce beam size for consistency
                                language=None,
                                vad_filter=True,
                                word_timestamps=False,
                                condition_on_previous_text=False,  # Prevent context bleeding
                                no_speech_threshold=0.6  # Higher threshold for silence
                            )
                            
                            for segment in segments:
                                text = segment.text.strip()
                                if text and not self.is_duplicate(text):
                                    print(f"[{info.language}]: {text}")
                                    self.recent_transcripts.append(text)
                                    
                        except Exception as e:
                            print(f"Transcription error: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")

    def start(self):
        self.is_running = True
        threading.Thread(target=self.process_audio, daemon=True).start()
        print("🎙️  LIVE MULTILINGUAL TRANSCRIPTION (No Duplicates)\n")

        try:
            with sd.InputStream(
                channels=self.CHANNELS,
                samplerate=self.SAMPLE_RATE,
                callback=self.audio_callback,
                blocksize=2048,
                dtype=np.float32
            ):
                while self.is_running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Stopped.")
        finally:
            self.is_running = False

if __name__ == "__main__":
    LiveTranscriber().start()

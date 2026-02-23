import speech_recognition as sr
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # ensures consistent results

def recognize_and_detect():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 Speak something... (English, Hindi, Tamil, Punjabi)")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        # Step 1: Use Google STT (multilingual capable if set to "hi-IN", "ta-IN", etc.)
        text = recognizer.recognize_google(audio)  # default auto tries English
        print(f"✅ Raw Recognized: {text}")

        # Step 2: Detect language
        lang = detect(text)

        # Step 3: Map detected language
        lang_map = {
            "en": "English",
            "hi": "Hindi",
            "ta": "Tamil",
            "pa": "Punjabi"
        }

        lang_name = lang_map.get(lang, "Unknown")
        print(f"🌐 Detected Language: {lang_name} ({lang})")

    except sr.UnknownValueError:
        print("❌ Sorry, could not understand audio")
    except sr.RequestError as e:
        print(f"⚠️ API error: {e}")

if __name__ == "__main__":
    recognize_and_detect()

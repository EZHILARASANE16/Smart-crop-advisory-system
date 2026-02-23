import speech_recognition as sr

# Available languages
LANGUAGES = {
    "1": ("English (India)", "en-IN"),
    "2": ("Hindi", "hi-IN"),
    "3": ("Tamil", "ta-IN"),
    "4": ("Punjabi", "pa-IN"),
}

def choose_language():
    print("\n🌐 Select a language:")
    for key, (name, _) in LANGUAGES.items():
        print(f"{key}. {name}")
    
    choice = input("\nEnter choice number: ").strip()
    if choice in LANGUAGES:
        return LANGUAGES[choice][1], LANGUAGES[choice][0]
    else:
        print("❌ Invalid choice. Defaulting to English (India).")
        return "en-IN", "English (India)"

def recognize_speech(language_code, language_name):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"\n🎤 Speak now in {language_name}...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language=language_code)
        print(f"✅ Recognized ({language_name}): {text}")
    except sr.UnknownValueError:
        print("❌ Sorry, could not understand audio")
    except sr.RequestError as e:
        print(f"⚠️ API error: {e}")

if __name__ == "__main__":
    while True:
        lang_code, lang_name = choose_language()
        recognize_speech(lang_code, lang_name)
        cont = input("\nDo you want to continue? (y/n): ").strip().lower()
        if cont != "y":
            print("🛑 Exiting...")
            break

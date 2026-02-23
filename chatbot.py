import os
import requests

API_KEY = "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"  

print("API_KEY is:", API_KEY)  


API_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"  

def chat_with_groq(user_message):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": user_message}
        ]
    }

    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]

if __name__ == "__main__":
    print("Groq AI Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        try:
            reply = chat_with_groq(user_input)
            print("Chatbot:", reply)
        except Exception as e:
            print("Error:", e)

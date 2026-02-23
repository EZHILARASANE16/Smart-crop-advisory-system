from flask import Flask, request, jsonify
import joblib
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import io
from chatbot import chat_with_groq
from faster_whisper import WhisperModel
import os, time
from selenium import webdriver
from selenium.webdriver.common.by import By
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ---------- Load Crop & Fertilizer Models ----------
try:
    crop_model = joblib.load("models/crop_model.pkl")
    crop_encoder = joblib.load("models/crop_label_encode.pkl")
    fertilizer_encoder = joblib.load("models/fertilizer_label_encode.pkl")
    fertilizer_model = joblib.load("models/fertilizer_model.pkl")
    print("✅ Crop & Fertilizer models loaded successfully")
except Exception as e:
    print("❌ Error loading crop/fertilizer models:", e)

# ---------- Load Disease Model ----------
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_healthy"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    disease_model = models.mobilenet_v2(pretrained=False)
    disease_model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.6),
        torch.nn.Linear(disease_model.last_channel, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.6),
        torch.nn.Linear(256, 128),
        torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.6),
        torch.nn.Linear(128, len(CLASS_NAMES))
    )
    checkpoint = torch.load("models/disease_model_best_fixed.pth", map_location=device)
    disease_model.load_state_dict(checkpoint["model_state_dict"])
    disease_model.to(device)
    disease_model.eval()
    disease_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    print("✅ Disease model loaded successfully")
except Exception as e:
    print("❌ Error loading disease model:", e)

# ---------- Whisper STT ----------
whisper_model = WhisperModel("small", device="cpu")  # switch to "cuda" if GPU available


# ---------- ROUTES ----------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "🚀 SIH Backend is running",
        "endpoints": ["/transcribe", "/predict_crop", "/recommend_fertilizer", "/predict_disease", "/chat", "/market_price"]
    })


# 1. Transcribe Audio (Whisper)
@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_file = request.files["audio"]
    filepath = "temp.wav"
    audio_file.save(filepath)

    segments, info = whisper_model.transcribe(filepath, beam_size=1)
    transcript = " ".join([seg.text for seg in segments])
    return jsonify({"transcript": transcript, "language": info.language})


# 2. Crop Prediction
@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    try:
        data = request.json
        features = np.array([[data["N"], data["P"], data["K"],
                              data["temperature"], data["humidity"],
                              data["ph"], data["rainfall"]]])
        prediction = crop_model.predict(features)[0]
        crop_name = crop_encoder.inverse_transform([prediction])[0]
        return jsonify({"predicted_crop": crop_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# 3. Fertilizer Recommendation
@app.route("/recommend_fertilizer", methods=["POST"])
def recommend_fertilizer():
    try:
        data = request.json
        features = np.array([[data["N"], data["P"], data["K"],
                              data["soil_type"], data["crop_type"]]])
        fert_pred = fertilizer_model.predict(features)[0]
        fert_name = fertilizer_encoder.inverse_transform([fert_pred])[0]
        return jsonify({"recommended_fertilizer": fert_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# 4. Disease Detection
@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"].read()
    img = Image.open(io.BytesIO(img_file)).convert("RGB")
    tensor = disease_preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = disease_model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)

    return jsonify({
        "predicted_disease": CLASS_NAMES[top_idx],
        "confidence": float(top_prob.item())
    })


# 5. Chatbot (Groq API)
@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_msg = request.json.get("message", "")
        reply = chat_with_groq(user_msg)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 6. Market Price (Selenium Scraping)
@app.route("/market_price", methods=["GET"])
def market_price():
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)

        driver.get("https://enam.gov.in/web/dashboard/trade-data")
        time.sleep(5)
        rows = driver.find_elements(By.XPATH, "//table//tr")

        headers = [th.text.strip() for th in rows[0].find_elements(By.TAG_NAME, "th")]
        results = []
        for row in rows[1:]:
            cols = [c.text.strip() for c in row.find_elements(By.TAG_NAME, "td")]
            if cols:
                results.append(dict(zip(headers, cols)))

        driver.quit()
        return jsonify({"market_prices": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- Run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

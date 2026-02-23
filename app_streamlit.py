# ============================
# app_streamlit.py
# ============================
# Run:
#   streamlit run app_streamlit.py

import os
import pandas as pd
import numpy as np
import streamlit as st
from joblib import load

# ============================
# Paths
# ============================
CROP_MODEL_PATH = os.path.join("models","crop_model.pkl")
CROP_LE_PATH = os.path.join("models","crop_label_encode.pkl")
FERT_MODEL_PATH = os.path.join("models","fertilizer_model.pkl")
FERT_LE_PATH = os.path.join("models","fertilizer_label_encode.pkl")
SOIL_PATH = os.path.join("data","soil_profiles_demo.csv")

# ============================
# Streamlit Config
# ============================
st.set_page_config(page_title="Smart Agri ML", layout="centered")
st.title("🌾 Smart Agri ML — Crop & Fertilizer Recommendation (Prototype)")

# ============================
# Loaders
# ============================
@st.cache_data
def load_soil_profiles(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_resource
def load_models():
    if not (os.path.exists(CROP_MODEL_PATH) and os.path.exists(CROP_LE_PATH)):
        st.error("❌ Crop model not found. Run `python train_crop_model.py` first.")
        st.stop()
    if not (os.path.exists(FERT_MODEL_PATH) and os.path.exists(FERT_LE_PATH)):
        st.error("❌ Fertilizer model not found. Run `python train_fertilizer.py` first.")
        st.stop()

    crop_model = load(CROP_MODEL_PATH)
    crop_le = load(CROP_LE_PATH)
    fert_model = load(FERT_MODEL_PATH)
    fert_encoders = load(FERT_LE_PATH)
    return crop_model, crop_le, fert_model, fert_encoders

soil_df = load_soil_profiles(SOIL_PATH)
crop_model, crop_le, fert_model, fert_encoders = load_models()

# ============================
# Prediction Helpers
# ============================
def predict_crop(features):
    cols = ["N","P","K","temperature","humidity","ph","rainfall"]
    X = np.array([features[c] for c in cols], dtype=float).reshape(1, -1)

    if hasattr(crop_model, "predict_proba"):
        probs = crop_model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = crop_le.inverse_transform([pred_idx])[0]
    else:
        pred_idx = int(crop_model.predict(X)[0])
        pred_label = crop_le.inverse_transform([pred_idx])[0]
        probs = None

    return pred_label, probs

def predict_fertilizer(features):
    X = pd.DataFrame([features])
    pred = fert_model.predict(X)[0]
    fert_name = fert_encoders['Fertilizer Name'].inverse_transform([pred])[0]
    return fert_name

# ============================
# Sidebar Navigation
# ============================
st.sidebar.header("Mode")
mode = st.sidebar.radio("Choose Recommendation:", ["Crop Recommendation", "Fertilizer Recommendation"])

# ============================
# Crop Recommendation
# ============================
if mode == "Crop Recommendation":
    input_mode = st.radio("Input method:", ["Manual inputs", "Use Location Data (demo)"])

    if input_mode.startswith("Manual"):
        st.subheader("🌱 Manual Crop Inputs")
        N = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0, 1.0)
        P = st.number_input("Phosphorus (P)", 0.0, 200.0, 35.0, 1.0)
        K = st.number_input("Potassium (K)", 0.0, 200.0, 60.0, 1.0)
        ph = st.number_input("Soil pH", 3.5, 9.5, 6.8, 0.1)
        temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 28.0, 0.5)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 65.0, 1.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 80.0, 1.0)

    else:
        st.subheader("📍 Location-based Crop Inputs")
        if soil_df.empty:
            st.warning("No soil profiles found. Add data/soil_profiles_demo.csv")
        else:
            states = sorted(soil_df["state"].unique())
            sel_state = st.selectbox("State", states, index=0)
            districts = sorted(soil_df[soil_df["state"]==sel_state]["district"].unique())
            sel_dist = st.selectbox("District", districts, index=0)

            row = soil_df[(soil_df["state"]==sel_state) & (soil_df["district"]==sel_dist)].iloc[0]

            st.json(row.to_dict())

            N = float(row["N"]); P = float(row["P"]); K = float(row["K"])
            ph = float(row["ph"]); temperature = float(row["temperature"])
            humidity = float(row["humidity"]); rainfall = float(row["rainfall"])

    if st.button("🔮 Predict Crop"):
        features = {"N":N, "P":P, "K":K, "temperature":temperature,
                    "humidity":humidity, "ph":ph, "rainfall":rainfall}
        pred_label, probs = predict_crop(features)
        st.success(f"✅ Recommended Crop: **{pred_label}**")

        if probs is not None:
            top_indices = np.argsort(probs)[::-1][:5]
            top_items = [(crop_le.inverse_transform([i])[0], float(probs[i])) for i in top_indices]
            st.subheader("Confidence (Top 5)")
            st.table(pd.DataFrame([{"Crop": k, "Probability": round(v,4)} for k,v in top_items]))

# ============================
# Fertilizer Recommendation
# ============================
elif mode == "Fertilizer Recommendation":
    input_mode = st.radio("Input method:", ["Manual inputs", "Use Location Data (demo)"])

    if input_mode.startswith("Manual"):
        st.subheader("🧪 Manual Fertilizer Inputs")
        temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 26.0, 0.5)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 52.0, 1.0)
        moisture = st.number_input("Moisture (%)", 0.0, 100.0, 40.0, 1.0)

        soil_types = list(fert_encoders['Soil Type'].classes_)
        crop_types = list(fert_encoders['Crop Type'].classes_)
        soil_type = st.selectbox("Soil Type", soil_types)
        crop_type = st.selectbox("Crop Type", crop_types)

        nitrogen = st.number_input("Nitrogen", 0.0, 200.0, 37.0, 1.0)
        potassium = st.number_input("Potassium", 0.0, 200.0, 0.0, 1.0)
        phosphorous = st.number_input("Phosphorous", 0.0, 200.0, 0.0, 1.0)

    else:
        st.subheader("📍 Location-based Fertilizer Inputs")
        if soil_df.empty:
            st.warning("No soil profiles found. Add data/soil_profiles_demo.csv")
        else:
            states = sorted(soil_df["state"].unique())
            sel_state = st.selectbox("State", states, index=0)
            districts = sorted(soil_df[soil_df["state"]==sel_state]["district"].unique())
            sel_dist = st.selectbox("District", districts, index=0)

            row = soil_df[(soil_df["state"]==sel_state) & (soil_df["district"]==sel_dist)].iloc[0]

            st.json(row.to_dict())

            temperature = float(row["temperature"])
            humidity = float(row["humidity"])
            moisture = float(row["humidity"]) * 0.6  # approx moisture
            nitrogen = float(row["N"])
            potassium = float(row["K"])
            phosphorous = float(row["P"])

            soil_types = list(fert_encoders['Soil Type'].classes_)
            crop_types = list(fert_encoders['Crop Type'].classes_)
            soil_type = st.selectbox("Soil Type", soil_types)
            crop_type = st.selectbox("Crop Type", crop_types)

    if st.button("🧪 Recommend Fertilizer"):
        features = {
            "Temparature": temperature,
            "Humidity ": humidity,
            "Moisture": moisture,
            "Soil Type": fert_encoders['Soil Type'].transform([soil_type])[0],
            "Crop Type": fert_encoders['Crop Type'].transform([crop_type])[0],
            "Nitrogen": nitrogen,
            "Potassium": potassium,
            "Phosphorous": phosphorous
        }
        fert_name = predict_fertilizer(features)
        st.success(f"✅ Recommended Fertilizer: **{fert_name}**")

st.markdown("---")
st.caption("Prototype: Demo soil profiles → Replace with real data (Soil Health Cards, Govt APIs).")

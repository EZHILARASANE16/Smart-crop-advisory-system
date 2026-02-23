# ============================
# train_crop_model.py
# ============================
# Usage:
#   1) Place your dataset file at: data/crop_recommendation.csv
#      Expected columns (Kaggle format):
#         N,P,K,temperature,humidity,ph,rainfall,label
#   2) Run: python train_crop_model.py
#   3) It will produce: models/crop_model.pkl and models/label_encoder.pkl

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier  # optional
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

DATA_PATH = os.path.join("data", "crop_recommendation.csv")
OUT_DIR = os.path.join("models")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Please add crop_recommendation.csv with {FEATURES + [TARGET]} columns."
        )

    df = pd.read_csv(DATA_PATH)
    for col in FEATURES + [TARGET]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    X = df[FEATURES].values
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest model
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"✅ Test Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, preds, target_names=le.classes_))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))

    dump(model, os.path.join(OUT_DIR, "crop_model.pkl"))
    dump(le, os.path.join(OUT_DIR, "label_encoder.pkl"))
    print("💾 Saved: models/crop_model.pkl, models/label_encoder.pkl")

if __name__ == "__main__":
    main()

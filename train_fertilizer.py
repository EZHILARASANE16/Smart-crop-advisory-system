# train_fertilizer.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import os

# 1️⃣ Load dataset
data_path = r"C:\Users\Godwin Arulraj\Desktop\sih2025\data\fertilizer_dataset.csv"
data = pd.read_csv(data_path)

print("🔍 Columns found:", data.columns)
print(data.head())

# 2️⃣ Encode categorical features
categorical_cols = ['Soil Type', 'Crop Type']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Encode target column 'Fertilizer Name'
target_col = 'Fertilizer Name'
target_le = LabelEncoder()
data[target_col] = target_le.fit_transform(data[target_col])
label_encoders[target_col] = target_le

# 3️⃣ Split features & target
X = data.drop(target_col, axis=1)
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Train Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# 5️⃣ Evaluate
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Random Forest Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_le.classes_))

# 6️⃣ Save model + encoders using joblib
model_dir = r"C:\Users\Godwin Arulraj\Desktop\sih2025\models"
os.makedirs(model_dir, exist_ok=True)

dump(rf_model, os.path.join(model_dir, "fertilizer_model.pkl"))
dump(label_encoders, os.path.join(model_dir, "fertilizer_label_encode.pkl"))

print("🎉 Fertilizer model + encoders saved successfully (joblib).")

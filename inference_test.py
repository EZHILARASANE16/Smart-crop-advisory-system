# inference_test.py
# ====================
# Load your best model checkpoint and run inference on images in a dedicated folder.

import os
import torch
from torchvision import transforms, models
from PIL import Image

# 1. Paths and configuration
MODEL_PATH = r"C:\Users\Godwin Arulraj\Desktop\sih2025\models\disease_model_best_fixed.pth"
TEST_DIR  = r"C:\Users\Godwin Arulraj\Desktop\sih2025\test_images"  # See 'Where to save' below
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Class names (must match training order)
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

# 3. Build the model architecture (match training)
model = models.mobilenet_v2(pretrained=False)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.6),
    torch.nn.Linear(model.last_channel, 256),
    torch.nn.BatchNorm1d(256),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(0.6),
    torch.nn.Linear(256, 128),
    torch.nn.BatchNorm1d(128),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(0.6),
    torch.nn.Linear(128, len(CLASS_NAMES))
)

# 4. Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# 5. Preprocessing (must match training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 6. Inference helper
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)
    return CLASS_NAMES[top_idx], top_prob.item()

# 7. Run inference on all images in TEST_DIR
for fname in os.listdir(TEST_DIR):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    path = os.path.join(TEST_DIR, fname)
    label, confidence = predict(path)
    print(f"{fname} → {label} ({confidence*100:.1f}%)")

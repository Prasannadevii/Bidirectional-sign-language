# main.py
import os
import cv2
import torch
import numpy as np
from torchvision import models
from fuzzy_features import compute_fuzzy_features
from feature_extraction import extract_hand_angles_from_frame

# --- Define Hybrid Model ---
import torch.nn as nn
import torch.nn.functional as F

class HybridSignModel(nn.Module):
    def __init__(self, num_classes=26, fuzzy_input_size=11):
        super(HybridSignModel, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.cnn_features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fuzzy_fc = nn.Linear(fuzzy_input_size, 64)
        self.fc1 = nn.Linear(1280 + 64, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, image, fuzzy_features):
        x1 = self.cnn_features(image)
        x1 = self.pool(x1)
        x1 = x1.view(x1.size(0), -1)
        x2 = F.relu(self.fuzzy_fc(fuzzy_features))
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Load Model ---
num_classes = 26
fuzzy_input_size = 11
model_path = "hybrid_model.pth"

if not os.path.exists(model_path):
    print("❌ hybrid_model.pth not found. Train the model first!")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridSignModel(num_classes=num_classes, fuzzy_input_size=fuzzy_input_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Define Labels ---
labels = [chr(i+65) for i in range(num_classes)]  # A-Z

# --- Start Camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Camera opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Extract Fuzzy Features ---
    angles = extract_hand_angles_from_frame(frame)
    fuzzy_features = compute_fuzzy_features(angles)
    fuzzy_tensor = torch.tensor(fuzzy_features, dtype=torch.float32).unsqueeze(0).to(device)

    # --- Prepare Image for CNN ---
    img = cv2.resize(frame, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = np.transpose(img, (2,0,1))
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    # --- Prediction ---
    with torch.no_grad():
        output = model(img_tensor, fuzzy_tensor)
        pred = torch.argmax(output, dim=1).item()

    # --- Display ---
    cv2.putText(frame, f"Predicted: {labels[pred]}", (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

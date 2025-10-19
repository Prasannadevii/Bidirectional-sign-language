# train_hybrid.py
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
from feature_extraction import extract_hand_angles_from_frame
from fuzzy_features import compute_fuzzy_features

# --- Hybrid Model ---
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

# --- Dataset Path ---
train_path = './data/asl_alphabet_train'  # Adjust path
train_folders = [os.path.join(train_path, f) for f in sorted(os.listdir(train_path))]

X_images, X_fuzzy, y_labels = [], [], []

print("📦 Loading images and computing fuzzy features...")

for class_idx, folder in enumerate(train_folders):
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (224,224))
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized = img_resized / 255.0
        img_resized = np.transpose(img_resized, (2,0,1))
        X_images.append(img_resized)

        angles = extract_hand_angles_from_frame(img)
        fuzzy = compute_fuzzy_features(angles)
        X_fuzzy.append(fuzzy)

        y_labels.append(class_idx)

X_images = torch.tensor(np.array(X_images), dtype=torch.float32)
X_fuzzy = torch.tensor(np.array(X_fuzzy), dtype=torch.float32)
y_labels = torch.tensor(y_labels, dtype=torch.long)

print(f"✅ Loaded {len(X_images)} samples.")

# --- Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridSignModel(num_classes=len(train_folders), fuzzy_input_size=11).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
batch_size = 16

print("🚀 Starting training...")

for epoch in range(epochs):
    running_loss = 0.0
    for i in range(0, len(X_images), batch_size):
        imgs = X_images[i:i+batch_size].to(device)
        fuzzy = X_fuzzy[i:i+batch_size].to(device)
        labels = y_labels[i:i+batch_size].to(device)

        optimizer.zero_grad()
        outputs = model(imgs, fuzzy)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(X_images):.4f}")

# --- Save Model ---
torch.save(model.state_dict(), 'hybrid_model.pth')
print("✅ Training complete. Model saved as hybrid_model.pth")

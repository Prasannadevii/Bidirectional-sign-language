import torch
import torch.nn as nn
import torchvision.models as models

class HybridSignModel(nn.Module):
    def __init__(self, num_classes=29, fuzzy_input_size=11):
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
        x2 = nn.functional.relu(self.fuzzy_fc(fuzzy_features))
        x = torch.cat([x1, x2], dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

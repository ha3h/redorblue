import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Settings
IMAGE_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rebuild the model (same as in training)
class ColorClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Load model
model = ColorClassifier().to(DEVICE)
model.load_state_dict(torch.load("red_blue_model.pth", map_location=DEVICE))
model.eval()

# Load your test image
img = Image.open("test_blue.jpg").convert("RGB")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

img_tensor = transform(img).unsqueeze(0).to(DEVICE)

# Make prediction
with torch.no_grad():
    output = model(img_tensor)
    prediction = "RED" if output.item() < 0.5 else "BLUE"
    print(f"Prediction: {prediction} ({output.item():.2f})")

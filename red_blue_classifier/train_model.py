import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Settings
IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load images and convert to tensors
dataset = datasets.ImageFolder(
    "dataset",
    transform=transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
)

print("Class labels:", dataset.class_to_idx)  # Should show {'0_red': 0, '1_blue': 1}

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Step 2: Build a simple neural network
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

model = ColorClassifier().to(DEVICE)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 3: Train the model
print("Training...")
for epoch in range(EPOCHS):
    total_loss = 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).float().unsqueeze(1)
        
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

# Step 4: Save the model
torch.save(model.state_dict(), "red_blue_model.pth")
print("Model saved!")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

class StegoCNN(nn.Module):
    def __init__(self):
        super(StegoCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# Load data
cover_data = datasets.ImageFolder("dataset", transform=transform)
train_loader = DataLoader(cover_data, batch_size=16, shuffle=True)

model = StegoCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5):
    for imgs, labels in train_loader:
        preds = model(imgs)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

#save model
torch.save(model.state_dict(), "stegocnn.pth")
print("Model saved as stegocnn.pth")

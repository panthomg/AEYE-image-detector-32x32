import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
# Check Me Model 0.1A

# --- CONFIGURATION --- data_dir = "/kaggle/input/deepfake-vs-real-60k/deepfake-vs-real-60k"

DATA_PATH = r"C:\Users\user\Downloads\archive" 
BATCH_SIZE = 32 
EPOCHS = 5 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. PREPARE IMAGES
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_set = datasets.ImageFolder(root=f"{DATA_PATH}/train", transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

# 2. LOAD PRE-MADE BRAIN (EfficientNet)
model = models.efficientnet_b0(weights='DEFAULT')
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2) # Real or Fake
model = model.to(DEVICE)

# 3. SET LEARNING RULES
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. START TRAINING
print(f"Training started on {DEVICE}...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1} finished. Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "ai_detector.pth")
print("Training complete! Brain saved as ai_detector.pth")

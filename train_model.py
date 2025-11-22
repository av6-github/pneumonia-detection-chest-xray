import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import numpy as np

# --- 1. SETUP ---
print("--- STEP 1: INITIALIZING ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {device}")

# Download Data
dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
base_dir = os.path.join(dataset_path, 'chest_xray')
if not os.path.exists(os.path.join(base_dir, 'train')):
    base_dir = os.path.join(dataset_path, 'chest_xray', 'chest_xray')

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# --- 2. PREPROCESSING ---
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15), # Increased rotation to force learning shapes
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Scanning dataset...")
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# --- 3. THE MAGIC FIX: WEIGHTED SAMPLER ---
# This section calculates how many Normal vs Pneumonia images exist
# and forces the loader to pick them equally.

targets = np.array(train_data.targets) # [0, 0, 1, 1, 1, 0...]
class_counts = np.bincount(targets)    # [Normal_Count, Pneumonia_Count]
print(f"Dataset Imbalance found: Normal={class_counts[0]}, Pneumonia={class_counts[1]}")

# Calculate weight for each class (inverse of count)
class_weights = 1. / class_counts
sample_weights = class_weights[targets] # Assign a weight to every single image

# Create the sampler
sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights).double(),
    num_samples=len(sample_weights),
    replacement=True
)

# Pass sampler to DataLoader (Note: shuffle=False is required when using sampler)
train_loader = DataLoader(train_data, batch_size=32, sampler=sampler, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, pin_memory=True)

# --- 4. MODEL ARCHITECTURE (ResNet18) ---
print("\n--- STEP 4: LOADING RESNET18 ---")
model = models.resnet18(weights='IMAGENET1K_V1') 

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1),
    nn.Sigmoid()
)

model = model.to(device)

# --- 5. TRAINING ---
criterion = nn.BCELoss() 
# Lower learning rate to prevent "forgetting" the pre-trained knowledge
optimizer = optim.Adam(model.parameters(), lr=0.00005) 

epochs = 5 # 5 Epochs ensures it sees enough "Normal" examples

print("\n--- STEP 5: BALANCED TRAINING STARTING ---")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    avg_loss = running_loss / len(train_loader)
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

print("\nSaving balanced model...")
torch.save(model.state_dict(), 'pneumonia_model.pth')
print("Done! Now run evaluate_model.py again.")
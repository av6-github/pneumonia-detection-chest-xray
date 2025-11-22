import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import numpy as np

# --- 1. SETUP ---
print("--- STEP 1: INITIALIZING HIGH-ACCURACY PIPELINE ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {device}")

# Download Data
dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
base_dir = os.path.join(dataset_path, 'chest_xray')
if not os.path.exists(os.path.join(base_dir, 'train')):
    base_dir = os.path.join(dataset_path, 'chest_xray', 'chest_xray')

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# --- 2. ADVANCED PREPROCESSING ---
# We add ColorJitter (contrast/brightness) and Zoom (RandomResizedCrop)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    # Simulate different X-Ray exposure levels
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    # Slight zoom to force looking at details
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# --- 3. BALANCED SAMPLER (Keep this, it's working!) ---
targets = np.array(train_data.targets)
class_counts = np.bincount(targets)
class_weights = 1. / class_counts
sample_weights = class_weights[targets]

sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights).double(),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_data, batch_size=32, sampler=sampler, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, pin_memory=True)

# --- 4. MODEL UPGRADE: DENSENET121 ---
print("\n--- STEP 4: LOADING DENSENET121 (Medical Standard) ---")
# DenseNet is more robust for medical images than ResNet
model = models.densenet121(weights='IMAGENET1K_V1') 

# DenseNet structure is different. The classifier is called 'classifier', not 'fc'
num_ftrs = model.classifier.in_features 
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 1),
    nn.Sigmoid()
)

model = model.to(device)

# --- 5. OPTIMIZER & SCHEDULER ---
criterion = nn.BCELoss()
# Start slightly faster (1e-4) because we have a scheduler now
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5) 

# SCHEDULER: "The Parking Sensor"
# Reduce learning rate by factor of 0.1 every 4 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

epochs = 10 # Increase epochs because we are finetuning deeper

print("\n--- STEP 5: TRAINING STARTING ---")
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
    
    # Update the learning rate
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    
    avg_loss = running_loss / len(train_loader)
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

print("\nSaving DenseNet model...")
torch.save(model.state_dict(), 'pneumonia_model.pth')
print("Done.")
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import kagglehub
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 5  # Enough to generate good-looking graphs
LEARNING_RATE = 0.0001

# --- 1. PREPARE DATA ---
print(">>> Downloading/Loading Data...")
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
base_dir = os.path.join(path, 'chest_xray')
if not os.path.exists(os.path.join(base_dir, 'train')):
    base_dir = os.path.join(path, 'chest_xray', 'chest_xray')

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test') # We use Test as Validation for graph purposes

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Data
train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

# Weighted Sampler (To fix imbalance)
targets = np.array(train_data.targets)
class_counts = np.bincount(targets)
class_weights = 1. / class_counts
sample_weights = class_weights[targets]
sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(), num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# --- 2. DEFINE MODEL (DenseNet121) ---
def get_model():
    model = models.densenet121(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    return model.to(DEVICE)

# --- 3. TRAINING LOOP WITH LOGGING ---
def train_engine():
    print(f">>> Starting Training on {DEVICE} for {EPOCHS} epochs...")
    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # History storage
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training Phase
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_epoch_loss = val_loss / len(test_loader)
        val_epoch_acc = val_correct / val_total
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")

    # Save Model
    torch.save(model.state_dict(), 'final_model.pth')
    return model, history

# --- 4. PLOTTING FUNCTIONS ---
def plot_training_curves(history):
    epochs_range = range(1, EPOCHS + 1)
    
    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('graph_accuracy.png')
    print("Saved: graph_accuracy.png")
    
    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('graph_loss.png')
    print("Saved: graph_loss.png")

def plot_confusion_matrix_and_roc(model):
    print(">>> Generating Confusion Matrix & ROC...")
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('graph_confusion_matrix.png')
    print("Saved: graph_confusion_matrix.png")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('graph_roc_curve.png')
    print("Saved: graph_roc_curve.png")
    
    # Print Text Report
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Train & Get History
    trained_model, history = train_engine()
    
    # 2. Plot Training Curves
    plot_training_curves(history)
    
    # 3. Plot Evaluation Metrics
    plot_confusion_matrix_and_roc(trained_model)
    
    print("\n>>> ALL GRAPHS GENERATED SUCCESSFULLY.")
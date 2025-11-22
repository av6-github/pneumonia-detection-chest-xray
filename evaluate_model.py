import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import kagglehub
import os

# --- HELPER: BUILD THE RESNET MODEL ---
def get_model():
    # We load the structure of ResNet18
    model = models.resnet18(weights=None) 
    
    # We change the last layer to match what we trained (1 output + Sigmoid)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    return model

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Evaluation on: {device}")
    
    # --- 1. LOAD DATA ---
    print("Locating Test Data...")
    try:
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        base_dir = os.path.join(path, 'chest_xray')
        if not os.path.exists(os.path.join(base_dir, 'test')):
            base_dir = os.path.join(path, 'chest_xray', 'chest_xray')
        test_dir = os.path.join(base_dir, 'test')
    except:
        print("Error: Internet required to locate dataset path.")
        return

    # Standard ResNet Normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_data = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    print(f"Classes found: {test_data.class_to_idx}") 

    # --- 2. LOAD MODEL ---
    model = get_model().to(device)
    
    if not os.path.exists('pneumonia_model.pth'):
        print("Error: 'pneumonia_model.pth' not found. Train the ResNet model first.")
        return
        
    try:
        model.load_state_dict(torch.load('pneumonia_model.pth', map_location=device))
        model.eval() # Freeze layers for testing
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Make sure you re-ran the training script with ResNet before running this.")
        return

    # --- 3. CALCULATE METRICS ---
    correct = 0
    total = 0
    tp, tn, fp, fn = 0, 0, 0, 0

    print("Running evaluation on 624 test images...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Confusion Matrix Stats
            tp += ((predicted == 1) & (labels == 1)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

    accuracy = 100 * correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # --- 4. PRINT REPORT ---
    print("\n" + "="*35)
    print("   FINAL PERFORMANCE REPORT")
    print("="*35)
    print(f"ACCURACY:  {accuracy:.2f}%  (Goal: >85%)")
    print(f"PRECISION: {precision:.2f}")
    print(f"RECALL:    {recall:.2f}")
    print(f"F1 SCORE:  {f1_score:.2f}")
    print("-" * 35)
    print("CONFUSION MATRIX:")
    print(f" [ TP ] Correctly Detected Pneumonia: {int(tp)}")
    print(f" [ TN ] Correctly Detected Healthy:   {int(tn)}")
    print(f" [ FP ] False Alarm (Healthy->Sick):  {int(fp)}")
    print(f" [ FN ] MISSED DIAGNOSIS (Sick->Ok):  {int(fn)}")
    print("="*35)

if __name__ == "__main__":
    evaluate()
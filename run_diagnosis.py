import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from torchvision import models

# --- CONFIGURATION ---
MODEL_PATH = 'pneumonia_model.pth'

# --- MODEL DEFINITION ---
def load_model():
    model = models.resnet18(weights=None) # Structure only
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    return model

def run_simple_diagnosis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    model = load_model().to(device) # Use the function instead of PneumoniaNet()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        print("Model file not found. Please train first.")
        return

    # Open File Selector
    root = tk.Tk()
    root.withdraw()
    img_path = filedialog.askopenfilename(title="Select X-Ray", filetypes=[("Images", "*.jpg *.jpeg *.png")])
    if not img_path: return

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        prob = output.item() # This is the raw number (0.0 to 1.0)

    # Logic: Closer to 0 is Normal, Closer to 1 is Pneumonia
    prediction = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    
    # --- OUTPUT ---
    print("\n" + "="*30)
    print("      DIAGNOSIS RESULT")
    print("="*30)
    print(f"PREDICTION:  {prediction}")
    print(f"PROBABILITY: {prob:.4f}")
    print("-" * 30)
    print("Explanation of Probability:")
    print("  0.00 - 0.50 -> NORMAL")
    print("  0.51 - 1.00 -> PNEUMONIA")
    print("="*30)

    # Visualization
    plt.imshow(image)
    plt.title(f"{prediction}\nProb: {prob:.4f}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    run_simple_diagnosis()
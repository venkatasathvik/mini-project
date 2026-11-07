import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np

# 1. HARDWARE-AWARE MODEL
class OpenPulseCNN(nn.Module):
    def __init__(self):
        super(OpenPulseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16384, 1, bias=False) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 2. DATA PIPELINE (Strictly scaled to 0-127 for hardware)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(), # Converts to 0.0 - 1.0
    transforms.Lambda(lambda x: x * 127.0) # Scales strictly to 0.0 - 127.0
])

print("Loading dataset...")
dataset = ImageFolder(root='dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
print(f"Found {len(dataset)} tiles. Classes: {dataset.class_to_idx}")

# 3. TRAINING LOOP
model = OpenPulseCNN()
criterion = nn.BCEWithLogitsLoss() # Best for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
print("Starting training...")
for epoch in range(epochs):
    epoch_loss = 0.0
    for images, labels in dataloader:
        # Reshape labels for BCE loss
        labels = labels.unsqueeze(1).float() 
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")

# 4. SYMMETRIC QUANTIZATION & COE EXPORT
def export_quantized_coe(tensor_fp32, filename):
    numpy_weights = tensor_fp32.detach().numpy()
    max_abs_val = np.max(np.abs(numpy_weights))
    
    if max_abs_val == 0:
        quantized = np.zeros_like(numpy_weights, dtype=np.int8)
    else:
        scale = 127.0 / max_abs_val
        quantized = np.round(numpy_weights * scale)
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
    
    flat = quantized.flatten()
    with open(filename, 'w') as f:
        f.write("memory_initialization_radix=16;\nmemory_initialization_vector=\n")
        for i, val in enumerate(flat):
            hex_val = f"{val & 0xFF:02X}"
            terminator = ";" if i == len(flat) - 1 else ",\n"
            f.write(f"{hex_val}{terminator}")
    print(f"Exported {filename} ({len(flat)} weights)")

print("Quantizing and exporting hardware files...")
export_quantized_coe(model.conv1.weight, 'conv1_weights.coe')
export_quantized_coe(model.fc.weight, 'fc_weights.coe')
print("Pipeline complete. Ready for Vivado.")

print("Quantizing and exporting hardware files...")

# Update these two lines to route the .coe files to your specific folder
export_quantized_coe(model.conv1.weight, 'project/output files/conv1_weights.coe')
export_quantized_coe(model.fc.weight, 'project/output files/fc_weights.coe')

print("Pipeline complete. Ready for Vivado.")
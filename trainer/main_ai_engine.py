import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np

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

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 127.0)
])

print("Loading dataset...")
dataset = ImageFolder(root='dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = OpenPulseCNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
print("Starting training...")
for epoch in range(epochs):
    epoch_loss = 0.0
    for images, labels in dataloader:
        labels = labels.unsqueeze(1).float() 
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")

print("\nEvaluating FP32 Training Accuracy...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in dataloader:
        outputs = model(images)
        predictions = (outputs > 0).squeeze().long()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
print(f"FP32 Training Accuracy: {(correct / total) * 100:.2f}%")

print("\nEvaluating Model on Unseen SkyPulse Data...")
try:
    test_dataset = ImageFolder(root='test_dataset_of_skypulse', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = model(images)
            predictions = (outputs > 0).long().view(-1)
            labels = labels.long().view(-1)
            correct_test += (predictions == labels).sum().item()
            total_test += labels.size(0)
            
    if total_test > 0:
        print(f"Unseen Data (SkyPulse) Accuracy: {(correct_test / total_test) * 100:.2f}%\n")
    else:
        print("No test data found. Sort tiles into test_dataset_of_skypulse/clear and cloudy.\n")
except Exception as e:
    print(f"Skipping test data evaluation: {e}\n")

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
            hex_val = f"{int(val) & 0xFF:02X}"
            terminator = ";" if i == len(flat) - 1 else ",\n"
            f.write(f"{hex_val}{terminator}")
    print(f"Exported {filename} ({len(flat)} weights)")

print("Quantizing and exporting hardware files...")
# Routing to the updated 'output_coe_files' folder
export_quantized_coe(model.conv1.weight, 'output_coe_files/conv1_weights.coe')
export_quantized_coe(model.fc.weight, 'output_coe_files/fc_weights.coe')
print("Pipeline complete. Ready for Vivado.")
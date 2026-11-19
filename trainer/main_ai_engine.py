import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

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

def train_stage(model, loader, criterion, optimizer, epochs, stage_name):
    model.train()
    print(f"\n--- {stage_name} ---")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in loader:
            labels = labels.unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f}")

def evaluate(model, loader, name, class_names):
    model.eval()
    correct, total = 0, 0
    class_correct = [0, 0]
    class_total = [0, 0]

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            preds = (outputs > 0).long().view(-1)
            labels = labels.long().view(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for c in (0, 1):
                mask = (labels == c)
                class_total[c] += mask.sum().item()
                class_correct[c] += (preds[mask] == labels[mask]).sum().item()

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    print(f"{name} Accuracy: {acc:.2f}% ({correct}/{total})")
    for c in (0, 1):
        if class_total[c] > 0:
            print(f"  - {class_names[c]}: {(class_correct[c]/class_total[c])*100:.2f}% ({class_correct[c]}/{class_total[c]})")
    return acc

print("Loading datasets...")
landsat_dataset = ImageFolder(root='dataset', transform=transform)
landsat_loader = DataLoader(landsat_dataset, batch_size=16, shuffle=True)

fy2h_train_dataset = ImageFolder(root='dataset_fy2h_train', transform=transform)
fy2h_train_loader = DataLoader(fy2h_train_dataset, batch_size=16, shuffle=True)

model = OpenPulseCNN()
criterion = nn.BCEWithLogitsLoss()

# Stage 1: Pretrain on Landsat 8 (38-Cloud)
optimizer_s1 = optim.Adam(model.parameters(), lr=0.001)
train_stage(model, landsat_loader, criterion, optimizer_s1, epochs=5, stage_name="Stage 1: Landsat 8 Pretraining")

# Stage 2: Domain-Adapt on FY-2H (Feb 1 - Apr 20)
optimizer_s2 = optim.Adam(model.parameters(), lr=0.0005)
train_stage(model, fy2h_train_loader, criterion, optimizer_s2, epochs=10, stage_name="Stage 2: FY-2H Domain Adaptation (Feb-Apr 20)")

print("\nEvaluating Training Accuracy (FY-2H Feb-Apr 20)...")
evaluate(model, fy2h_train_loader, "FY-2H Train", fy2h_train_dataset.classes)

print("\nEvaluating Model on Unseen SkyPulse Data (Apr 21 - May 20)...")
test_dataset = ImageFolder(root='test_dataset_of_skypulse', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
evaluate(model, test_loader, "Unseen SkyPulse (Apr 21-May 20)", test_dataset.classes)

# Step 7: CPU Software Latency Benchmark
print("\nBenchmarking CPU Software Inference Latency...")
model.eval()
sample_img, _ = test_dataset[0]
sample_batch = sample_img.unsqueeze(0)  # Shape: (1, 1, 128, 128)

with torch.no_grad():
    for _ in range(20):  # CPU cache warmup
        _ = model(sample_batch)

    iterations = 500
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = model(sample_batch)
    t1 = time.perf_counter()

avg_ms = ((t1 - t0) / iterations) * 1000.0
fps = 1000.0 / avg_ms
print(f"CPU Software Latency (128x128, FP32): {avg_ms:.4f} ms/image | Throughput: {fps:.1f} FPS")

def export_quantized_coe(tensor_fp32, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
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

print("\nQuantizing and exporting hardware files...")
export_quantized_coe(model.conv1.weight, 'output_coe_files/conv1_weights.coe')
export_quantized_coe(model.fc.weight, 'output_coe_files/fc_weights.coe')
print("Pipeline complete. Ready for Vivado.")
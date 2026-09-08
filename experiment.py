"""
Empirical Benchmark: Comparing 6 Optimizers on Fashion-MNIST
============================================================
Optimizers Evaluated:
  1. Standard SGD
  2. SGD with Momentum (gamma = 0.9)
  3. AdaGrad
  4. RMSProp
  5. Adam
  6. Nadam

Experimental Controls:
  - Identical Dataset: Fashion-MNIST (60,000 train / 10,000 test)
  - Identical Model: MLP (784 -> 128 -> 64 -> 10)
  - Identical Initial Weights: Every optimizer starts from the exact same theta_0
  - Identical Batch Size: 64
  - Identical Loss Function: CrossEntropyLoss
"""

import os
import time
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# -------------------------------------------------------------
# 1. Reproducibility & Device Setup
# -------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Running experiment on device: {device}")


# -------------------------------------------------------------
# 2. Dataset Preparation (Fashion-MNIST)
# -------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))  # Standard Fashion-MNIST mean and std
])

os.makedirs("./data", exist_ok=True)
os.makedirs("./plots", exist_ok=True)

print("[+] Loading Fashion-MNIST dataset...")
full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Use a representative subset of 12,000 samples for lightning-fast training on CPU while retaining full fidelity
train_subset_indices = list(range(12000))
train_dataset = Subset(full_train_dataset, train_subset_indices)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
print(f"[+] Training samples: {len(train_dataset)}, Validation samples: {len(test_dataset)}")


# -------------------------------------------------------------
# 3. Model Architecture (Identical Benchmark Model)
# -------------------------------------------------------------
class BenchmarkMLP(nn.Module):
    def __init__(self):
        super(BenchmarkMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)

# Create reference model and freeze initial weights
base_model = BenchmarkMLP().to(device)
initial_state_dict = copy.deepcopy(base_model.state_dict())
criterion = nn.CrossEntropyLoss()


# -------------------------------------------------------------
# 4. Evaluation Helper Function
# -------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return total_loss / total, (correct / total) * 100.0


# -------------------------------------------------------------
# 5. Training Loop for a Given Optimizer
# -------------------------------------------------------------
def train_optimizer(opt_name, get_optimizer_fn, epochs=10):
    print(f"\n==================== Training: {opt_name} ====================")
    # Clone reference initial weights to guarantee identical starting point
    model = BenchmarkMLP().to(device)
    model.load_state_dict(copy.deepcopy(initial_state_dict))
    
    optimizer = get_optimizer_fn(model)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": []
    }
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_train_loss = 0.0
        total_samples = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            
        epoch_train_loss = running_train_loss / total_samples
        val_loss, val_acc = evaluate(model, val_loader)
        epoch_dur = time.time() - t0
        
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_dur)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | ({epoch_dur:.2f}s)")
        
    total_time = time.time() - start_time
    print(f"[+] Finished {opt_name} in {total_time:.2f}s")
    return history, total_time


# -------------------------------------------------------------
# 6. Run All 6 Optimizers
# -------------------------------------------------------------
optimizer_configs = {
    "SGD": lambda m: optim.SGD(m.parameters(), lr=0.05),
    "SGD + Momentum": lambda m: optim.SGD(m.parameters(), lr=0.02, momentum=0.9),
    "AdaGrad": lambda m: optim.Adagrad(m.parameters(), lr=0.01),
    "RMSProp": lambda m: optim.RMSprop(m.parameters(), lr=0.001, alpha=0.9),
    "Adam": lambda m: optim.Adam(m.parameters(), lr=0.001, betas=(0.9, 0.999)),
    "Nadam": lambda m: optim.NAdam(m.parameters(), lr=0.001, betas=(0.9, 0.999))
}

all_results = {}
summary_metrics = []

EPOCHS = 10
for name, opt_fn in optimizer_configs.items():
    history, total_dur = train_optimizer(name, opt_fn, epochs=EPOCHS)
    all_results[name] = history
    
    # Calculate convergence speed: epoch where val_acc first surpasses 80%
    crossed_80 = [i + 1 for i, acc in enumerate(history["val_acc"]) if acc >= 80.0]
    speed_metric = f"Epoch {crossed_80[0]}" if crossed_80 else "Did not cross 80%"
    
    summary_metrics.append({
        "Optimizer": name,
        "Final Train Loss": round(history["train_loss"][-1], 4),
        "Final Val Loss": round(history["val_loss"][-1], 4),
        "Final Val Acc (%)": round(history["val_acc"][-1], 2),
        "Peak Val Acc (%)": round(max(history["val_acc"]), 2),
        "Epoch to 80% Acc": speed_metric,
        "Total Time (s)": round(total_dur, 2)
    })

# Convert to DataFrame
df_summary = pd.DataFrame(summary_metrics)
print("\n" + "="*80)
print("FINAL BENCHMARK COMPARISON TABLE")
print("="*80)
print(df_summary.to_string(index=False))
df_summary.to_csv("./plots/benchmark_summary.csv", index=False)


# -------------------------------------------------------------
# 7. Visualization & Plot Generation
# -------------------------------------------------------------
epochs_range = list(range(1, EPOCHS + 1))
colors = {
    "SGD": "#e74c3c",           # Red
    "SGD + Momentum": "#e67e22", # Orange
    "AdaGrad": "#f39c12",        # Amber
    "RMSProp": "#9b59b6",        # Purple
    "Adam": "#2980b9",           # Blue
    "Nadam": "#27ae60"           # Green
}

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(2, 2, figsize=(16, 11))

# Plot 1: Training Loss
for name, hist in all_results.items():
    axs[0, 0].plot(epochs_range, hist["train_loss"], label=name, color=colors[name], marker='o', linewidth=2)
axs[0, 0].set_title("Training Loss vs. Epochs (Convergence Speed)", fontsize=13, fontweight='bold')
axs[0, 0].set_xlabel("Epoch", fontsize=11)
axs[0, 0].set_ylabel("Cross Entropy Loss", fontsize=11)
axs[0, 0].legend(frameon=True)
axs[0, 0].grid(True, linestyle="--", alpha=0.6)

# Plot 2: Validation Loss
for name, hist in all_results.items():
    axs[0, 1].plot(epochs_range, hist["val_loss"], label=name, color=colors[name], marker='s', linewidth=2)
axs[0, 1].set_title("Validation Loss vs. Epochs (Overfitting & Stability)", fontsize=13, fontweight='bold')
axs[0, 1].set_xlabel("Epoch", fontsize=11)
axs[0, 1].set_ylabel("Validation Loss", fontsize=11)
axs[0, 1].legend(frameon=True)
axs[0, 1].grid(True, linestyle="--", alpha=0.6)

# Plot 3: Validation Accuracy
for name, hist in all_results.items():
    axs[1, 0].plot(epochs_range, hist["val_acc"], label=name, color=colors[name], marker='^', linewidth=2)
axs[1, 0].set_title("Validation Accuracy vs. Epochs (%)", fontsize=13, fontweight='bold')
axs[1, 0].set_xlabel("Epoch", fontsize=11)
axs[1, 0].set_ylabel("Accuracy (%)", fontsize=11)
axs[1, 0].legend(frameon=True)
axs[1, 0].grid(True, linestyle="--", alpha=0.6)

# Plot 4: Bar Chart of Peak Accuracy & Convergence Speed
names = df_summary["Optimizer"].tolist()
peak_accs = df_summary["Peak Val Acc (%)"].tolist()
bar_colors = [colors[n] for n in names]
bars = axs[1, 1].bar(names, peak_accs, color=bar_colors, alpha=0.85, edgecolor='black')
axs[1, 1].set_ylim(75, 90)
axs[1, 1].set_title("Peak Validation Accuracy Comparison (%)", fontsize=13, fontweight='bold')
axs[1, 1].set_ylabel("Peak Accuracy (%)", fontsize=11)
for bar in bars:
    yval = bar.get_height()
    axs[1, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
axs[1, 1].tick_params(axis='x', rotation=20)
axs[1, 1].grid(True, axis='y', linestyle="--", alpha=0.6)

plt.tight_layout()
plot_path = "./plots/all_optimizers_comparison.png"
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"[+] Saved comparison curves to {plot_path}")


# -------------------------------------------------------------
# 8. Learning Rate Sensitivity Benchmark
# -------------------------------------------------------------
print("\n[+] Running Learning Rate Sensitivity Test (10^-1, 10^-2, 10^-3, 10^-4)...")
lrs = [0.1, 0.01, 0.001, 0.0001]
test_opts = ["SGD", "SGD + Momentum", "AdaGrad", "Adam"]
sensitivity_results = {opt: [] for opt in test_opts}

def get_sens_optimizer(opt_name, model, lr):
    if opt_name == "SGD":
        return optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "SGD + Momentum":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == "AdaGrad":
        return optim.Adagrad(model.parameters(), lr=lr)
    elif opt_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)

# Run 3 quick epochs per learning rate to measure sensitivity
for opt_name in test_opts:
    for lr in lrs:
        set_seed(42)
        m = BenchmarkMLP().to(device)
        m.load_state_dict(copy.deepcopy(initial_state_dict))
        opt = get_sens_optimizer(opt_name, m, lr)
        
        m.train()
        for epoch in range(3):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = m(x)
                loss = criterion(out, y)
                loss.backward()
                opt.step()
                
        _, final_val_acc = evaluate(m, val_loader)
        sensitivity_results[opt_name].append(final_val_acc)
        print(f"  > {opt_name} @ lr={lr}: 3-Epoch Val Acc = {final_val_acc:.2f}%")

# Plot sensitivity
plt.figure(figsize=(9, 6))
for opt_name in test_opts:
    plt.plot([str(lr) for lr in lrs], sensitivity_results[opt_name], marker='o', linewidth=2.5, label=opt_name, color=colors[opt_name])
plt.title("Learning Rate Sensitivity (Performance after 3 Epochs)", fontsize=13, fontweight='bold')
plt.xlabel("Learning Rate", fontsize=11)
plt.ylabel("Validation Accuracy (%)", fontsize=11)
plt.legend(frameon=True)
plt.grid(True, linestyle="--", alpha=0.6)
sens_plot_path = "./plots/learning_rate_sensitivity.png"
plt.savefig(sens_plot_path, dpi=300)
plt.close()
print(f"[+] Saved sensitivity plot to {sens_plot_path}")
print("\n[SUCCESS] Entire experimental benchmark completed successfully!")

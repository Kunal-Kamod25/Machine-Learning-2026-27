"""
Generate a comprehensive, beautifully structured Jupyter Notebook:
optimizers_deep_dive.ipynb
"""

import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(source_text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_text.strip().split("\n")]
    })

def add_code(source_text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_text.strip().split("\n")]
    })

# -------------------------------------------------------------
# Section 1: Title & Group Overview
# -------------------------------------------------------------
add_md("""# Deep Dive into Deep Learning Optimizers
### A Comprehensive Empirical & Theoretical Benchmark: SGD, Momentum, AdaGrad, RMSProp, Adam & Nadam

**Authors (Group of 3):**
* **Person 1:** Foundations of Optimization, Standard SGD & SGD with Momentum
* **Person 2:** The Adaptive Revolution: AdaGrad & RMSProp
* **Person 3:** Modern Hybrids: Adam, Nadam & The Generalization Mystery
* **Collaborative:** Empirical Benchmark, Learning-Rate Sensitivity & Viva Defense

---

### Notebook Structure
1. **The Foundations:** Loss Landscape, Gradients, Step Size & Parameter Updates
2. **Optimizer Mechanics & Math:** 12-Point Analysis for all 6 Optimizers
3. **Reproducible Experiment Setup:** Fashion-MNIST & Identical MLP Architecture
4. **Benchmark Execution:** Training all 6 Optimizers under Identical Conditions
5. **Empirical Visualizations:** Training Loss, Validation Loss, Accuracy & Peak Acc
6. **Learning-Rate Sensitivity Test:** Analyzing Stability across 4 Decades ($10^{-1}$ to $10^{-4}$)
7. **The Generalization Mystery:** Why Fast Optimization $\\neq$ Best Model Quality
8. **Viva & Oral Defense Cheat Sheet:** 4-Tier Answers to Real-World Questions
""")

# -------------------------------------------------------------
# Section 2: Foundations & Formulas
# -------------------------------------------------------------
add_md("""## 1. The Foundational Theory

### 1.1 What is Training a Neural Network?
A neural network has millions of adjustable parameters (weights and biases $\\theta$).
We feed an input $x$, generate prediction $\\hat{y}$, and compute loss $L(\\theta)$.
The goal of optimization is to find $\\theta^*$ that minimizes $L(\\theta)$.

### 1.2 The Gradient $\\nabla_\\theta L(\\theta)$
The gradient is a vector containing all partial derivatives of the loss function:
$$\\nabla_\\theta L(\\theta) = \\left[ \\frac{\\partial L}{\\partial \\theta_1}, \\frac{\\partial L}{\\partial \\theta_2}, \\dots, \\frac{\\partial L}{\\partial \\theta_d} \\right]^T$$
* **Direction:** Points in the direction of **steepest uphill ascent**.
* **Descent:** To minimize loss, we step in the **opposite direction**: $-\\nabla_\\theta L(\\theta)$.

### 1.3 The Baseline Weight Update Rule
$$\\theta_{t+1} = \\theta_t - \\eta \\cdot g_t$$
where $\\eta$ is the learning rate (step size), and $g_t = \\nabla_\\theta L(\\theta_t)$.
""")

# -------------------------------------------------------------
# Section 3: 6 Optimizers Breakdown
# -------------------------------------------------------------
add_md("""## 2. Summary Comparison of the 6 Optimizers

| Optimizer | 1st Moment (Direction) | 2nd Moment (Adaptive Scale) | Update Equation | Analogy |
| :--- | :---: | :---: | :--- | :--- |
| **SGD** | None | None | $\\theta_{t+1} = \\theta_t - \\eta g_t$ | Blind hiker taking quick, noisy steps |
| **SGD + Momentum** | Exponential Average ($\\gamma=0.9$) | None | $v_{t+1} = \\gamma v_t + \\eta g_t$<br>$\\theta_{t+1} = \\theta_t - v_{t+1}$ | Heavy bowling ball rolling down a valley |
| **AdaGrad** | None | Cumulative Sum ($G_t = \\sum g^2$) | $\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{G_t + \\epsilon}} g_t$ | An elephant remembering every past mistake |
| **RMSProp** | None | Leaky Average ($\\beta=0.9$) | $v_t = \\beta v_{t-1} + (1-\\beta)g_t^2$<br>$\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{v_t + \\epsilon}} g_t$ | Practical human focusing on recent performance |
| **Adam** | Leaky Average ($\\beta_1=0.9$) | Leaky Average ($\\beta_2=0.999$) | $\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$ | High-speed vehicle with active suspension |
| **Nadam** | Nesterov Lookahead | Leaky Average ($\\beta_2=0.999$) | $\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\bar{m}_t$ | Racecar driver braking before entering a curve |
""")

# -------------------------------------------------------------
# Section 4: Environment & Setup Code
# -------------------------------------------------------------
add_md("""## 3. Experimental Setup & Reproducibility Controls

To ensure 100% scientific fairness:
1. **Identical Initial Weights:** Every single optimizer starts from the exact same cloned weight parameters $\\theta_0$.
2. **Identical Seed:** Fixed random seed (42) for batch ordering and initialization.
3. **Identical Architecture:** 3-layer MLP ($784 \\to 128 \\to 64 \\to 10$) with ReLU activations.
4. **Identical Dataset:** Fashion-MNIST (10 clothing classes).
""")

add_code("""# Import required libraries
import os
import copy
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 1. Set global random seed for 100% reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 2. Configure computing device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[OK] Training environment configured on device: {device}")
""")

# -------------------------------------------------------------
# Section 5: Data Loading & Preprocessing
# -------------------------------------------------------------
add_md("""## 4. Dataset Loading: Fashion-MNIST

Fashion-MNIST consists of $28 \\times 28$ grayscale images across 10 clothing categories.
It is significantly more realistic and non-convex than standard digit MNIST.
""")

add_code("""# Normalize images using standard Fashion-MNIST mean and standard deviation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# Download dataset (stored locally in ./data)
train_dataset_full = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Use a clean 12,000 sample subset for fast and responsive training in notebooks
train_indices = list(range(12000))
train_subset = Subset(train_dataset_full, train_indices)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

print(f"[OK] Fashion-MNIST ready: {len(train_subset)} training samples, {len(test_dataset)} validation samples.")
""")

# -------------------------------------------------------------
# Section 6: Model Definition
# -------------------------------------------------------------
add_md("""## 5. Neural Network Architecture

We construct a multi-layer perceptron (MLP) with two hidden layers and ReLU non-linearities:
$$\\text{Input } (28 \\times 28 = 784) \\longrightarrow \\text{Dense}(128) \\longrightarrow \\text{ReLU} \\longrightarrow \\text{Dense}(64) \\longrightarrow \\text{ReLU} \\longrightarrow \\text{Dense}(10)$$
""")

add_code("""class BenchmarkMLP(nn.Module):
    \"\"\"Identical neural network architecture evaluated across all 6 optimizers.\"\"\"
    def __init__(self):
        super(BenchmarkMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)

# Create a reference model and clone its exact initial weights
# This guarantees that EVERY optimizer begins from the identical starting point theta_0!
reference_model = BenchmarkMLP().to(device)
initial_weights = copy.deepcopy(reference_model.state_dict())
criterion = nn.CrossEntropyLoss()

print("[OK] Reference model created and initial weights snapshot saved.")
""")

# -------------------------------------------------------------
# Section 7: Training & Evaluation Functions
# -------------------------------------------------------------
add_md("""## 6. Training & Evaluation Engine

Here we define the evaluation function and the isolated training loop for any optimizer.
""")

add_code("""def evaluate_model(model, data_loader):
    \"\"\"Compute validation loss and classification accuracy.\"\"\"
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += images.size(0)
            
    avg_loss = total_loss / total_samples
    accuracy_pct = (correct_predictions / total_samples) * 100.0
    return avg_loss, accuracy_pct


def train_single_optimizer(optimizer_name, get_optimizer_fn, epochs=10):
    \"\"\"
    Trains an independent model instance from initial_weights using the specified optimizer.
    Records epoch-by-epoch loss, validation accuracy, and execution time.
    \"\"\"
    print(f\"\\n>>> Training with: {optimizer_name} <<<\")
    
    # Reset model to identical starting weights
    model = BenchmarkMLP().to(device)
    model.load_state_dict(copy.deepcopy(initial_weights))
    
    optimizer = get_optimizer_fn(model)
    
    history = {
        \"train_loss\": [],
        \"val_loss\": [],
        \"val_acc\": [],
        \"epoch_times\": []
    }
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0
        samples_count = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)
            samples_count += images.size(0)
            
        epoch_train_loss = running_train_loss / samples_count
        val_loss, val_acc = evaluate_model(model, val_loader)
        epoch_duration = time.time() - epoch_start
        
        history[\"train_loss\"].append(epoch_train_loss)
        history[\"val_loss\"].append(val_loss)
        history[\"val_acc\"].append(val_acc)
        history[\"epoch_times\"].append(epoch_duration)
        
        print(f\"  Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | ({epoch_duration:.2f}s)\")
        
    total_duration = time.time() - start_time
    print(f\"[Done] {optimizer_name} completed in {total_duration:.2f} seconds.\")
    return history, total_duration
""")

# -------------------------------------------------------------
# Section 8: Benchmark Execution Across All 6
# -------------------------------------------------------------
add_md("""## 7. Running the Benchmark Across All 6 Optimizers

We instantiate each optimizer with its standard, academically accepted hyperparameter defaults:
* **SGD:** $\\eta = 0.05$
* **SGD + Momentum:** $\\eta = 0.02, \\gamma = 0.9$
* **AdaGrad:** $\\eta = 0.01$
* **RMSProp:** $\\eta = 0.001, \\alpha = 0.9$
* **Adam:** $\\eta = 0.001, \\beta_1 = 0.9, \\beta_2 = 0.999$
* **Nadam:** $\\eta = 0.001, \\beta_1 = 0.9, \\beta_2 = 0.999$
""")

add_code("""# Define the 6 optimizer factory functions
optimizer_dict = {
    \"SGD\": lambda m: optim.SGD(m.parameters(), lr=0.05),
    \"SGD + Momentum\": lambda m: optim.SGD(m.parameters(), lr=0.02, momentum=0.9),
    \"AdaGrad\": lambda m: optim.Adagrad(m.parameters(), lr=0.01),
    \"RMSProp\": lambda m: optim.RMSprop(m.parameters(), lr=0.001, alpha=0.9),
    \"Adam\": lambda m: optim.Adam(m.parameters(), lr=0.001, betas=(0.9, 0.999)),
    \"Nadam\": lambda m: optim.NAdam(m.parameters(), lr=0.001, betas=(0.9, 0.999))
}

all_histories = {}
summary_rows = []
TOTAL_EPOCHS = 10

# Execute training for all 6 optimizers sequentially
for opt_name, factory_fn in optimizer_dict.items():
    hist, duration = train_single_optimizer(opt_name, factory_fn, epochs=TOTAL_EPOCHS)
    all_histories[opt_name] = hist
    
    # Calculate convergence speed: epoch where validation accuracy first crossed 80%
    crossed_80 = [i + 1 for i, acc in enumerate(hist[\"val_acc\"]) if acc >= 80.0]
    speed = f\"Epoch {crossed_80[0]}\" if crossed_80 else \"Did not cross 80%\"
    
    summary_rows.append({
        \"Optimizer\": opt_name,
        \"Final Train Loss\": round(hist[\"train_loss\"][-1], 4),
        \"Final Val Loss\": round(hist[\"val_loss\"][-1], 4),
        \"Final Val Acc (%)\": round(hist[\"val_acc\"][-1], 2),
        \"Peak Val Acc (%)\": round(max(hist[\"val_acc\"]), 2),
        \"Epoch to 80% Acc\": speed,
        \"Total Time (s)\": round(duration, 2)
    })

# Render final summary comparison DataFrame
df_results = pd.DataFrame(summary_rows)
print(\"\\n\" + \"=\"*85)
print(\"FINAL OPTIMIZER BENCHMARK COMPARISON TABLE\")
print(\"=\"*85)
df_results
""")

# -------------------------------------------------------------
# Section 9: Visualization Plots
# -------------------------------------------------------------
add_md("""## 8. Comparative Visualizations

We generate a 4-panel visual comparison displaying:
1. **Training Loss:** Optimization speed and convergence rate.
2. **Validation Loss:** Overfitting and generalization behavior.
3. **Validation Accuracy:** Practical classification quality over time.
4. **Peak Validation Accuracy:** Final performance comparison.
""")

add_code("""epochs_axis = list(range(1, TOTAL_EPOCHS + 1))
color_map = {
    \"SGD\": \"#e74c3c\",
    \"SGD + Momentum\": \"#e67e22\",
    \"AdaGrad\": \"#f39c12\",
    \"RMSProp\": \"#9b59b6\",
    \"Adam\": \"#2980b9\",
    \"Nadam\": \"#27ae60\"
}

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Subplot 1: Training Loss
for name, hist in all_histories.items():
    axs[0, 0].plot(epochs_axis, hist[\"train_loss\"], label=name, color=color_map[name], marker='o', linewidth=2)
axs[0, 0].set_title(\"Training Loss vs. Epochs (Convergence Speed)\", fontsize=12, fontweight='bold')
axs[0, 0].set_xlabel(\"Epoch\")
axs[0, 0].set_ylabel(\"Cross Entropy Loss\")
axs[0, 0].legend()
axs[0, 0].grid(True, linestyle=\"--\", alpha=0.6)

# Subplot 2: Validation Loss
for name, hist in all_histories.items():
    axs[0, 1].plot(epochs_axis, hist[\"val_loss\"], label=name, color=color_map[name], marker='s', linewidth=2)
axs[0, 1].set_title(\"Validation Loss vs. Epochs (Generalization)\", fontsize=12, fontweight='bold')
axs[0, 1].set_xlabel(\"Epoch\")
axs[0, 1].set_ylabel(\"Validation Loss\")
axs[0, 1].legend()
axs[0, 1].grid(True, linestyle=\"--\", alpha=0.6)

# Subplot 3: Validation Accuracy
for name, hist in all_histories.items():
    axs[1, 0].plot(epochs_axis, hist[\"val_acc\"], label=name, color=color_map[name], marker='^', linewidth=2)
axs[1, 0].set_title(\"Validation Accuracy vs. Epochs (%)\", fontsize=12, fontweight='bold')
axs[1, 0].set_xlabel(\"Epoch\")
axs[1, 0].set_ylabel(\"Accuracy (%)\")
axs[1, 0].legend()
axs[1, 0].grid(True, linestyle=\"--\", alpha=0.6)

# Subplot 4: Peak Validation Accuracy Bar Chart
opts = df_results[\"Optimizer\"].tolist()
peaks = df_results[\"Peak Val Acc (%)\"].tolist()
bars = axs[1, 1].bar(opts, peaks, color=[color_map[o] for o in opts], alpha=0.85, edgecolor='black')
axs[1, 1].set_ylim(75, 90)
axs[1, 1].set_title(\"Peak Validation Accuracy (%)\", fontsize=12, fontweight='bold')
axs[1, 1].set_ylabel(\"Peak Accuracy (%)\")
for bar in bars:
    y = bar.get_height()
    axs[1, 1].text(bar.get_x() + bar.get_width()/2.0, y + 0.3, f\"{y:.2f}%\", ha='center', va='bottom', fontweight='bold')
axs[1, 1].tick_params(axis='x', rotation=15)
axs[1, 1].grid(True, axis='y', linestyle=\"--\", alpha=0.6)

plt.tight_layout()
plt.show()
""")

# -------------------------------------------------------------
# Section 10: Learning Rate Sensitivity
# -------------------------------------------------------------
add_md("""## 9. Learning Rate Sensitivity Benchmark

A critical engineering question: **How robust is each optimizer if we pick a suboptimal learning rate?**
We evaluate SGD, SGD+Momentum, AdaGrad, and Adam across four decades of learning rates:
$$\\eta \\in \\{0.1, 0.01, 0.001, 0.0001\\}$$
""")

add_code("""learning_rates = [0.1, 0.01, 0.001, 0.0001]
tested_optimizers = [\"SGD\", \"SGD + Momentum\", \"AdaGrad\", \"Adam\"]
sensitivity_scores = {opt: [] for opt in tested_optimizers}

def build_sensitivity_optimizer(opt_name, model_instance, lr_val):
    if opt_name == \"SGD\":
        return optim.SGD(model_instance.parameters(), lr=lr_val)
    elif opt_name == \"SGD + Momentum\":
        return optim.SGD(model_instance.parameters(), lr=lr_val, momentum=0.9)
    elif opt_name == \"AdaGrad\":
        return optim.Adagrad(model_instance.parameters(), lr=lr_val)
    elif opt_name == \"Adam\":
        return optim.Adam(model_instance.parameters(), lr=lr_val)

print(\"[+] Executing Learning Rate Sensitivity Sweeps (3 Epochs each)...\\n\")
for opt_name in tested_optimizers:
    for lr in learning_rates:
        set_seed(42)
        m = BenchmarkMLP().to(device)
        m.load_state_dict(copy.deepcopy(initial_weights))
        opt = build_sensitivity_optimizer(opt_name, m, lr)
        
        m.train()
        for epoch in range(3):
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                opt.zero_grad()
                out = m(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                opt.step()
                
        _, final_acc = evaluate_model(m, val_loader)
        sensitivity_scores[opt_name].append(final_acc)
        print(f\"  > {opt_name:15s} | lr={lr:<7} | 3-Epoch Val Acc: {final_acc:.2f}%\")

# Plot sensitivity curve
plt.figure(figsize=(10, 6))
for opt_name in tested_optimizers:
    plt.plot([str(lr) for lr in learning_rates], sensitivity_scores[opt_name], 
             marker='o', linewidth=2.5, label=opt_name, color=color_map[opt_name])

plt.title(\"Learning Rate Sensitivity: Validation Accuracy after 3 Epochs\", fontsize=13, fontweight='bold')
plt.xlabel(\"Learning Rate (\\u03b7)\", fontsize=11)
plt.ylabel(\"Validation Accuracy (%)\", fontsize=11)
plt.legend(frameon=True)
plt.grid(True, linestyle=\"--\", alpha=0.6)
plt.show()
""")

# -------------------------------------------------------------
# Section 11: Generalization & Deep Dive Discussion
# -------------------------------------------------------------
add_md("""## 10. The Generalization Mystery: Adam vs. SGD + Momentum

### Key Question:
> *"Why does Adam converge faster during training, yet SGD with Momentum often produces better generalization on unseen test data?"*

### The Answer: Flat vs. Sharp Minima
1. **Optimization Speed $\\neq$ Model Quality:** Fast drop in training loss does not guarantee superior performance on unseen distributions.
2. **Sharp Minima (Adam):** Because Adam adaptively scales coordinates, it easily slips into narrow, sharp crevices. In a sharp minimum, a tiny shift between training and test sets causes an exponential error surge.
3. **Flat Minima (SGD + Momentum):** Because SGD+Momentum uses uniform step sizes and kinetic velocity, it bounces out of sharp crevices and settles into broad, flat valleys. In a flat minimum, small distributional shifts produce virtually no change in test accuracy.

```
 Loss
  ▲
  │       Sharp Minimum (Adam)           Flat Minimum (SGD + Momentum)
  │       High Test Error Shift             Low Test Error Shift
  │
  │           \     /                              \             /
  │            \   /                                \___________/
  │             \_/
  │          Training: Low Loss               Training: Low Loss
  │          Testing:  HIGH ERROR             Testing:  LOW ERROR
  └───────────────────────────────────────────────────────────────────► Weights (θ)
```
""")

# -------------------------------------------------------------
# Section 12: Viva Q&A Defense
# -------------------------------------------------------------
add_md("""## 11. Viva Defense Quick Reference (10-Second Answers)

* **Q1: What optimizer for sparse gradients?**  
  *Answer:* **AdaGrad or Adam**, because their adaptive second moment divides by historical squared gradients, automatically amplifying updates for rare features.
* **Q2: What optimizer for noisy gradients?**  
  *Answer:* **SGD with Momentum or Adam**, because the first-moment running average acts as a low-pass filter, dampening noisy oscillations.
* **Q3: How to escape saddle points?**  
  *Answer:* **Adam or SGD + Momentum**, because accumulated velocity carries parameters across flat zero-gradient plateaus.
* **Q4: Why divide by $(1-\\beta^t)$ in Adam?**  
  *Answer:* Because moment vectors are initialized at zero; dividing by $(1-\\beta^t)$ cancels out the initialization bias during early iterations.
""")

# Write output file
output_path = "c:/Users/kunal/OneDrive/Desktop/Machine-Learning-2026-27/optimizers_deep_dive.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"[SUCCESS] Wrote notebook successfully to {output_path}")

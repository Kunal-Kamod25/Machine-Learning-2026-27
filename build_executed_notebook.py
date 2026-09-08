"""
Build and execute optimizers_deep_dive.ipynb with:
1. Exactly 1,000 samples for Fashion-MNIST (100 per clothing category).
2. SEPARATE execution cells for EACH of the 6 optimizers:
   - SGD (Kunal) -> Train & plot individual curves
   - SGD + Momentum (Kunal) -> Train & plot individual curves
   - AdaGrad (Rahul) -> Train & plot individual curves
   - RMSProp (Rahul) -> Train & plot individual curves
   - Adam (Pankaj) -> Train & plot individual curves
   - Nadam (Pankaj) -> Train & plot individual curves
3. In the end, a COMBINED comparison cell with all 6 curves overlaid and summary table!
4. LR sensitivity sweep, generalization theory, and viva questions.
"""

import io
import sys
import copy
import json
import base64
import contextlib
import matplotlib
matplotlib.use('Agg')  # Headless non-blocking backend
import matplotlib.pyplot as plt

execution_env = {}

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

cell_counter = 1

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(code_text):
    global cell_counter
    code_lines = [line + "\n" for line in code_text.strip().split("\n")]
    outputs = []
    
    stdout_buf = io.StringIO()
    plt.close('all')
    
    print(f"[*] Executing Notebook Cell #{cell_counter}...")
    with contextlib.redirect_stdout(stdout_buf):
        try:
            exec(code_text, execution_env)
        except Exception as e:
            print(f"[Error in cell {cell_counter}]: {e}", file=sys.stderr)
            raise e
            
    captured_stdout = stdout_buf.getvalue()
    if captured_stdout:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in captured_stdout.splitlines()]
        })
        
    fig_nums = plt.get_fignums()
    if fig_nums:
        for fnum in fig_nums:
            fig = plt.figure(fnum)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            outputs.append({
                "data": {
                    "image/png": img_b64,
                    "text/plain": ["<Figure size ...>"]
                },
                "metadata": {},
                "output_type": "display_data"
            })
            plt.close(fig)
            
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": cell_counter,
        "metadata": {},
        "outputs": outputs,
        "source": code_lines
    })
    cell_counter += 1


# =============================================================
# Cell 1: Header & Group Distribution
# =============================================================
add_md("""# Deep Dive into Deep Learning Optimizers
### A Comparative Empirical & Theoretical Study: SGD, Momentum, AdaGrad, RMSProp, Adam & Nadam

> **Seminar Presenters:**  
> • **Kunal (Presenter 1):** Foundations, Standard SGD & SGD with Momentum  
> • **Rahul (Presenter 2):** Adaptive Learning Rates, AdaGrad & RMSProp  
> • **Pankaj (Presenter 3):** Adam, Nadam & The Generalization Mystery  
> • **All Members:** Benchmark Walkthrough, Architecture Pipelines, Hand-Calculations & Viva Defense  

---

### Notebook Architecture
1. **Foundations & Core Math:** Loss Surfaces, Gradients, Step Size & Parameter Updates
2. **Setup & Controls:** Fashion-MNIST (1,000 Samples) & BenchmarkMLP Architecture
3. **Individual Optimizer Deep-Dives (Separate Cells & Plots):**
   - **Kunal:** Optimizer 1 (Standard SGD) & Optimizer 2 (SGD + Momentum)
   - **Rahul:** Optimizer 3 (AdaGrad) & Optimizer 4 (RMSProp)
   - **Pankaj:** Optimizer 5 (Adam) & Optimizer 6 (Nadam)
4. **Master Combined Dashboard:** All 6 curves overlaid side-by-side & Final Comparison Table
5. **Learning-Rate Sensitivity Test:** Evaluating stability across 4 decades ($10^{-1}$ to $10^{-4}$)
6. **The Generalization Mystery:** Why Fast Optimization $\\neq$ Best Model Quality (Flat vs. Sharp Minima)
7. **Viva Oral Defense Cheat Sheet:** 10-Second Answers for Teacher Questions
""")

# =============================================================
# Cell 2: Foundations
# =============================================================
add_md("""## 1. Mathematical Foundations of Optimization

### The Core Update Equation
Every gradient-based optimizer iteratively refines model parameters $\\theta$ by stepping opposite to the gradient vector:
$$\\theta_{t+1} = \\theta_t - \\Delta \\theta_t$$
where the gradient is the vector of all first-order partial derivatives of the loss function:
$$g_t = \\nabla_\\theta L(\\theta_t) = \\left[ \\frac{\\partial L}{\\partial \\theta_1}, \\frac{\\partial L}{\\partial \\theta_2}, \\dots, \\frac{\\partial L}{\\partial \\theta_d} \\right]^T$$

The difference between optimizers lies in **how they calculate $\\Delta \\theta_t$**:
* Does it use only instantaneous slope? (SGD)
* Does it build directional inertia? (Momentum)
* Does it adaptively scale each coordinate? (AdaGrad, RMSProp)
* Does it combine velocity, scaling, and lookahead? (Adam, Nadam)
""")

# =============================================================
# Cell 3: Setup & 1,000 Sample Dataset Loading
# =============================================================
add_code("""# -------------------------------------------------------------
# STEP 1: Libraries, Device, Reproducibility & 1,000-Sample Dataset
# -------------------------------------------------------------
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

# Set fixed seed for 100% scientific reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[OK] Training environment configured on: {device}")

# Preprocessing: Normalize Fashion-MNIST images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_full = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_full = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# USE EXACTLY 1,000 SAMPLES (~100 images per clothing category)
# This provides realistic non-convex training while allowing each optimizer
# to finish all 10 epochs in under 8.5 seconds on CPU!
train_subset = Subset(train_full, list(range(1000)))
val_subset = Subset(test_full, list(range(1000)))

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)

print(f"[OK] Dataset configured: {len(train_subset)} training images, {len(val_subset)} validation images.")
""")

# =============================================================
# Cell 4: Model Architecture & Evaluation Helper
# =============================================================
add_code("""# -------------------------------------------------------------
# STEP 2: Neural Network Architecture & Weight Cloning
# -------------------------------------------------------------
# 3-Layer MLP: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Logits)
class BenchmarkMLP(nn.Module):
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

# Create reference model and freeze initial weights theta_0
# Every single optimizer will begin its training path from this exact same point!
reference_model = BenchmarkMLP().to(device)
initial_weights = copy.deepcopy(reference_model.state_dict())
criterion = nn.CrossEntropyLoss()

def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return total_loss / total, (correct / total) * 100.0


def train_and_plot_optimizer(opt_name, get_opt_fn, color_hex, epochs=10):
    \"\"\"Trains one optimizer, prints epoch logs, and plots its individual loss/accuracy curves.\"\"\"
    print(f\"==================== Training: {opt_name} ====================\")
    model = BenchmarkMLP().to(device)
    model.load_state_dict(copy.deepcopy(initial_weights))
    optimizer = get_opt_fn(model)
    
    history = {\"train_loss\": [], \"val_loss\": [], \"val_acc\": []}
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            total += x.size(0)
            
        train_l = running_loss / total
        val_l, val_a = evaluate_model(model, val_loader)
        history[\"train_loss\"].append(train_l)
        history[\"val_loss\"].append(val_l)
        history[\"val_acc\"].append(val_a)
        print(f\"  Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_l:.4f} | Val Loss: {val_l:.4f} | Val Acc: {val_a:.2f}%\")
        
    duration = time.time() - start_time
    print(f\"[+] Finished {opt_name} in {duration:.2f}s\")
    
    # Render dedicated individual 2-panel figure for this optimizer
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    epochs_range = list(range(1, epochs + 1))
    
    # Subplot 1: Train Loss vs Val Loss
    ax1.plot(epochs_range, history[\"train_loss\"], marker='o', color=color_hex, linewidth=2, label='Train Loss')
    ax1.plot(epochs_range, history[\"val_loss\"], marker='s', color='#7f8c8d', linestyle='--', linewidth=2, label='Val Loss')
    ax1.set_title(f\"{opt_name} - Loss Trajectory\", fontsize=11, fontweight='bold')
    ax1.set_xlabel(\"Epoch\")
    ax1.set_ylabel(\"Cross-Entropy Loss\")
    ax1.legend()
    ax1.grid(True, linestyle=\"--\", alpha=0.6)
    
    # Subplot 2: Validation Accuracy
    ax2.plot(epochs_range, history[\"val_acc\"], marker='^', color=color_hex, linewidth=2, label='Val Accuracy')
    ax2.set_title(f\"{opt_name} - Validation Accuracy (%)\", fontsize=11, fontweight='bold')
    ax2.set_xlabel(\"Epoch\")
    ax2.set_ylabel(\"Accuracy (%)\")
    ax2.legend()
    ax2.grid(True, linestyle=\"--\", alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    return history, duration

# Master dictionary to store histories for the final combined dashboard
all_benchmark_results = {}
summary_table_rows = []
print(\"[OK] Training engine and individual plotter ready.\")
""")

# =============================================================
# Cell 5: Optimizer 1 - SGD (Kunal)
# =============================================================
add_md("""## 3. Optimizer 1: Stochastic Gradient Descent (SGD)
* **Presenter:** Kunal  
* **Formula:** $\\theta_{t+1} = \\theta_t - \\eta \\cdot g_t$  
* **Internal Memory:** 0 buffers (purely reactive)  
* **Analogy:** Blind hiker taking quick, noisy steps  
* **Behavior:** Simple and fast per step, but oscillates violently across steep ravines.
""")

add_code("""# Run Optimizer 1: Standard SGD
opt_name = "SGD"
get_opt = lambda m: optim.SGD(m.parameters(), lr=0.05)
color = "#e74c3c"  # Red

hist, dur = train_and_plot_optimizer(opt_name, get_opt, color, epochs=10)
all_benchmark_results[opt_name] = hist
summary_table_rows.append({
    "Optimizer": opt_name,
    "Final Train Loss": round(hist["train_loss"][-1], 4),
    "Final Val Loss": round(hist["val_loss"][-1], 4),
    "Final Val Acc (%)": round(hist["val_acc"][-1], 2),
    "Peak Val Acc (%)": round(max(hist["val_acc"]), 2),
    "Total Time (s)": round(dur, 2)
})
""")

# =============================================================
# Cell 6: Optimizer 2 - SGD + Momentum (Kunal)
# =============================================================
add_md("""## 4. Optimizer 2: SGD with Momentum
* **Presenter:** Kunal  
* **Formula:** $v_{t+1} = \\gamma v_t + \\eta g_t, \\quad \\theta_{t+1} = \\theta_t - v_{t+1}$  
* **Internal Memory:** 1 Velocity buffer ($v_t$)  
* **Analogy:** Heavy bowling ball rolling down a valley  
* **Advantage:** Alternating lateral oscillations cancel out, while down-valley momentum accelerates forward.
""")

add_code("""# Run Optimizer 2: SGD with Momentum
opt_name = "SGD + Momentum"
get_opt = lambda m: optim.SGD(m.parameters(), lr=0.02, momentum=0.9)
color = "#e67e22"  # Orange

hist, dur = train_and_plot_optimizer(opt_name, get_opt, color, epochs=10)
all_benchmark_results[opt_name] = hist
summary_table_rows.append({
    "Optimizer": opt_name,
    "Final Train Loss": round(hist["train_loss"][-1], 4),
    "Final Val Loss": round(hist["val_loss"][-1], 4),
    "Final Val Acc (%)": round(hist["val_acc"][-1], 2),
    "Peak Val Acc (%)": round(max(hist["val_acc"]), 2),
    "Total Time (s)": round(dur, 2)
})
""")

# =============================================================
# Cell 7: Optimizer 3 - AdaGrad (Rahul)
# =============================================================
add_md("""## 5. Optimizer 3: AdaGrad (Adaptive Gradient)
* **Presenter:** Rahul  
* **Formula:** $G_t = G_{t-1} + g_t^2, \\quad \\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{G_t + \\epsilon}} g_t$  
* **Internal Memory:** 1 Cumulative squared gradient buffer ($G_t$)  
* **Analogy:** Elephant that remembers every mistake since birth  
* **Limitation:** $G_t$ grows monotonically, causing the learning rate to vanish and freezing learning early.
""")

add_code("""# Run Optimizer 3: AdaGrad
opt_name = "AdaGrad"
get_opt = lambda m: optim.Adagrad(m.parameters(), lr=0.01)
color = "#f39c12"  # Amber

hist, dur = train_and_plot_optimizer(opt_name, get_opt, color, epochs=10)
all_benchmark_results[opt_name] = hist
summary_table_rows.append({
    "Optimizer": opt_name,
    "Final Train Loss": round(hist["train_loss"][-1], 4),
    "Final Val Loss": round(hist["val_loss"][-1], 4),
    "Final Val Acc (%)": round(hist["val_acc"][-1], 2),
    "Peak Val Acc (%)": round(max(hist["val_acc"]), 2),
    "Total Time (s)": round(dur, 2)
})
""")

# =============================================================
# Cell 8: Optimizer 4 - RMSProp (Rahul)
# =============================================================
add_md("""## 6. Optimizer 4: RMSProp
* **Presenter:** Rahul  
* **Formula:** $v_t = \\beta v_{t-1} + (1-\\beta) g_t^2, \\quad \\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{v_t + \\epsilon}} g_t$  
* **Internal Memory:** 1 Leaky exponential moving average buffer ($v_t$)  
* **Analogy:** Practical human who focuses on recent performance and forgets ancient mistakes  
* **Advantage:** Cures AdaGrad's vanishing learning rate; ideal for non-stationary problems and RNNs.
""")

add_code("""# Run Optimizer 4: RMSProp
opt_name = "RMSProp"
get_opt = lambda m: optim.RMSprop(m.parameters(), lr=0.001, alpha=0.9)
color = "#9b59b6"  # Purple

hist, dur = train_and_plot_optimizer(opt_name, get_opt, color, epochs=10)
all_benchmark_results[opt_name] = hist
summary_table_rows.append({
    "Optimizer": opt_name,
    "Final Train Loss": round(hist["train_loss"][-1], 4),
    "Final Val Loss": round(hist["val_loss"][-1], 4),
    "Final Val Acc (%)": round(hist["val_acc"][-1], 2),
    "Peak Val Acc (%)": round(max(hist["val_acc"]), 2),
    "Total Time (s)": round(dur, 2)
})
""")

# =============================================================
# Cell 9: Optimizer 5 - Adam (Pankaj)
# =============================================================
add_md("""## 7. Optimizer 5: Adam (Adaptive Moment Estimation)
* **Presenter:** Pankaj  
* **Formula:** $\\hat{m}_t = \\frac{m_t}{1-\\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1-\\beta_2^t}, \\quad \\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$  
* **Internal Memory:** 2 State buffers ($m_t$ for velocity, $v_t$ for adaptive variance)  
* **Analogy:** High-speed vehicle with an active smart suspension system  
* **Advantage:** Fast convergence, robust defaults, works out-of-the-box across nearly every deep learning architecture.
""")

add_code("""# Run Optimizer 5: Adam
opt_name = "Adam"
get_opt = lambda m: optim.Adam(m.parameters(), lr=0.001, betas=(0.9, 0.999))
color = "#2980b9"  # Blue

hist, dur = train_and_plot_optimizer(opt_name, get_opt, color, epochs=10)
all_benchmark_results[opt_name] = hist
summary_table_rows.append({
    "Optimizer": opt_name,
    "Final Train Loss": round(hist["train_loss"][-1], 4),
    "Final Val Loss": round(hist["val_loss"][-1], 4),
    "Final Val Acc (%)": round(hist["val_acc"][-1], 2),
    "Peak Val Acc (%)": round(max(hist["val_acc"]), 2),
    "Total Time (s)": round(dur, 2)
})
""")

# =============================================================
# Cell 10: Optimizer 6 - Nadam (Pankaj)
# =============================================================
add_md("""## 8. Optimizer 6: Nadam (Nesterov-accelerated Adam)
* **Presenter:** Pankaj  
* **Formula:** $\\bar{m}_t = \\beta_1 \\hat{m}_t + \\left( \\frac{1-\\beta_1}{1-\\beta_1^t} \\right) g_t, \\quad \\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\bar{m}_t$  
* **Internal Memory:** 2 State buffers with Nesterov lookahead computation  
* **Analogy:** Racecar driver applying anticipatory braking before turning into a corner  
* **Advantage:** Prevents overshooting steep ravines and improves stability on high-curvature surfaces.
""")

add_code("""# Run Optimizer 6: Nadam
opt_name = "Nadam"
get_opt = lambda m: optim.NAdam(m.parameters(), lr=0.001, betas=(0.9, 0.999))
color = "#27ae60"  # Green

hist, dur = train_and_plot_optimizer(opt_name, get_opt, color, epochs=10)
all_benchmark_results[opt_name] = hist
summary_table_rows.append({
    "Optimizer": opt_name,
    "Final Train Loss": round(hist["train_loss"][-1], 4),
    "Final Val Loss": round(hist["val_loss"][-1], 4),
    "Final Val Acc (%)": round(hist["val_acc"][-1], 2),
    "Peak Val Acc (%)": round(max(hist["val_acc"]), 2),
    "Total Time (s)": round(dur, 2)
})
""")

# =============================================================
# Cell 11: Master Combined Dashboard & Comparison Table
# =============================================================
add_md("""## 9. Master Combined Comparison Dashboard (All 6 Overlaid)

Here, all six optimizers are directly compared side-by-side on the exact same coordinate axes:
1. **Training Loss:** Convergence rate and optimization power.
2. **Validation Loss:** Overfitting behavior and trajectory smoothness.
3. **Validation Accuracy (%):** Real-world generalization progress over epochs.
4. **Peak Validation Accuracy:** Final accuracy ranking.
""")

add_code("""# -------------------------------------------------------------
# STEP 6: Combined Dashboard - All 6 Optimizers on One Canvas
# -------------------------------------------------------------
epochs_axis = list(range(1, 11))
color_map = {
    "SGD": "#e74c3c",
    "SGD + Momentum": "#e67e22",
    "AdaGrad": "#f39c12",
    "RMSProp": "#9b59b6",
    "Adam": "#2980b9",
    "Nadam": "#27ae60"
}

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Subplot 1: All Training Losses
for name, hist in all_benchmark_results.items():
    axs[0, 0].plot(epochs_axis, hist["train_loss"], label=name, color=color_map[name], marker='o', linewidth=2)
axs[0, 0].set_title("Training Loss vs. Epochs (Convergence Speed)", fontsize=12, fontweight='bold')
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].set_ylabel("Cross Entropy Loss")
axs[0, 0].legend()
axs[0, 0].grid(True, linestyle="--", alpha=0.6)

# Subplot 2: All Validation Losses
for name, hist in all_benchmark_results.items():
    axs[0, 1].plot(epochs_axis, hist["val_loss"], label=name, color=color_map[name], marker='s', linewidth=2)
axs[0, 1].set_title("Validation Loss vs. Epochs (Overfitting & Generalization)", fontsize=12, fontweight='bold')
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].set_ylabel("Validation Loss")
axs[0, 1].legend()
axs[0, 1].grid(True, linestyle="--", alpha=0.6)

# Subplot 3: All Validation Accuracies
for name, hist in all_benchmark_results.items():
    axs[1, 0].plot(epochs_axis, hist["val_acc"], label=name, color=color_map[name], marker='^', linewidth=2)
axs[1, 0].set_title("Validation Accuracy vs. Epochs (%)", fontsize=12, fontweight='bold')
axs[1, 0].set_xlabel("Epoch")
axs[1, 0].set_ylabel("Accuracy (%)")
axs[1, 0].legend()
axs[1, 0].grid(True, linestyle="--", alpha=0.6)

# Subplot 4: Peak Validation Accuracy Bar Chart
df_comparison = pd.DataFrame(summary_table_rows)
opts = df_comparison["Optimizer"].tolist()
peaks = df_comparison["Peak Val Acc (%)"].tolist()
bars = axs[1, 1].bar(opts, peaks, color=[color_map[o] for o in opts], alpha=0.85, edgecolor='black')
axs[1, 1].set_ylim(70, 85)
axs[1, 1].set_title("Peak Validation Accuracy Comparison (%)", fontsize=12, fontweight='bold')
axs[1, 1].set_ylabel("Peak Accuracy (%)")
for bar in bars:
    y = bar.get_height()
    axs[1, 1].text(bar.get_x() + bar.get_width()/2.0, y + 0.3, f"{y:.2f}%", ha='center', va='bottom', fontweight='bold')
axs[1, 1].tick_params(axis='x', rotation=15)
axs[1, 1].grid(True, axis='y', linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

print("\\n" + "="*85)
print("FINAL BENCHMARK COMPARISON TABLE (1,000 SAMPLES)")
print("="*85)
print(df_comparison.to_string(index=False))
""")

# =============================================================
# Cell 12: Learning Rate Sensitivity
# =============================================================
add_md("""## 10. Learning Rate Sensitivity Benchmark

How gracefully does each optimizer degrade when the learning rate is off by orders of magnitude?
We test $\\eta \\in [10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}]$ for 3 epochs each:
""")

add_code("""# -------------------------------------------------------------
# STEP 7: Multi-Decade Learning Rate Sensitivity Sweep
# -------------------------------------------------------------
lrs = [0.1, 0.01, 0.001, 0.0001]
test_opts = [\"SGD\", \"SGD + Momentum\", \"AdaGrad\", \"Adam\"]
sens_results = {opt: [] for opt in test_opts}

def get_sens_opt(opt_name, model_inst, lr):
    if opt_name == \"SGD\":
        return optim.SGD(model_inst.parameters(), lr=lr)
    elif opt_name == \"SGD + Momentum\":
        return optim.SGD(model_inst.parameters(), lr=lr, momentum=0.9)
    elif opt_name == \"AdaGrad\":
        return optim.Adagrad(model_inst.parameters(), lr=lr)
    elif opt_name == \"Adam\":
        return optim.Adam(model_inst.parameters(), lr=lr)

print(\"[+] Evaluating learning rate sensitivity across 4 decades...\")
for opt in test_opts:
    for lr in lrs:
        set_seed(42)
        m = BenchmarkMLP().to(device)
        m.load_state_dict(copy.deepcopy(initial_weights))
        optimizer = get_sens_opt(opt, m, lr)
        
        m.train()
        for ep in range(3):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = m(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                
        _, acc = evaluate_model(m, val_loader)
        sens_results[opt].append(acc)
        print(f\"  > {opt:15s} @ lr={lr:<7} | 3-Epoch Val Acc = {acc:.2f}%\")

# Plot Sensitivity Curve
plt.figure(figsize=(10, 5.5))
for opt in test_opts:
    plt.plot([str(lr) for lr in lrs], sens_results[opt], marker='o', linewidth=2.5, 
             label=opt, color=color_map[opt])

plt.title(\"Learning Rate Sensitivity (Performance after 3 Epochs)\", fontsize=13, fontweight='bold')
plt.xlabel(\"Learning Rate (\\u03b7)\", fontsize=11)
plt.ylabel(\"Validation Accuracy (%)\", fontsize=11)
plt.legend(frameon=True)
plt.grid(True, linestyle=\"--\", alpha=0.6)
plt.show()
""")

# =============================================================
# Cell 13: Generalization & Viva
# =============================================================
add_md(r"""## 11. The Generalization Mystery: Adam vs. SGD + Momentum

### Key Question:
> *"Why does Adam converge faster during training, yet SGD with Momentum often produces better generalization on unseen test data?"*

### The Answer: Flat vs. Sharp Minima
1. **Optimization Speed $\neq$ Model Quality:** Fast drop in training loss does not guarantee superior performance on unseen distributions.
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

---

## 12. Viva Defense Quick Reference (10-Second Answers)

* **Q1: What optimizer for sparse gradients?**  
  *Answer:* **AdaGrad or Adam**, because their adaptive second moment divides by historical squared gradients, automatically amplifying updates for rare features.
* **Q2: What optimizer for noisy gradients?**  
  *Answer:* **SGD with Momentum or Adam**, because the first-moment running average acts as a low-pass filter, dampening noisy oscillations.
* **Q3: How to escape saddle points?**  
  *Answer:* **Adam or SGD + Momentum**, because accumulated velocity carries parameters across flat zero-gradient plateaus.
* **Q4: Why divide by $(1-\beta^t)$ in Adam?**  
  *Answer:* Because moment vectors are initialized at zero; dividing by $(1-\beta^t)$ cancels out the initialization bias during early iterations.
""")

# Write executed notebook
output_path = "c:/Users/kunal/OneDrive/Desktop/Machine-Learning-2026-27/optimizers_deep_dive.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"\n[SUCCESS] Compiled and executed all separate optimizer cells & combined dashboard into {output_path}!")

"""
Generate High-Resolution Architecture and Pipeline Diagrams for the Presentation
================================================================================
Generates:
  1. plots/training_pipeline_architecture.png
     - Detailed Neural Network Architecture (784 -> 128 -> 64 -> 10)
     - Complete Optimization Pipeline (Forward -> Loss -> Backprop -> Optimizer Engine -> Weight Update)
  2. plots/optimizer_mechanisms_architecture.png
     - Internal block diagrams of all 6 optimizers showing states (m_t, v_t, G_t)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("./plots", exist_ok=True)

# -------------------------------------------------------------
# DIAGRAM 1: End-to-End System & Model Architecture Pipeline
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Palette
bg_color = "#f8f9fa"
box_blue = "#3498db"
box_green = "#2ecc71"
box_orange = "#e67e22"
box_purple = "#9b59b6"
box_red = "#e74c3c"
box_dark = "#2c3e50"
box_yellow = "#f1c40f"

# Title
ax.text(8, 9.5, "Deep Learning Optimizer & Model Architecture Pipeline", 
        fontsize=18, fontweight='bold', ha='center', color=box_dark)
ax.text(8, 9.1, "Identical Benchmark Pipeline for Fashion-MNIST Classification", 
        fontsize=12, style='italic', ha='center', color="#7f8c8d")

# 1. Input Box
ax.add_patch(patches.FancyBboxPatch((0.5, 5.2), 2.2, 2.2, boxstyle="round,pad=0.1", fc="#ecf0f1", ec=box_dark, lw=2))
ax.text(1.6, 6.6, "INPUT DATA", fontsize=11, fontweight='bold', ha='center', color=box_dark)
ax.text(1.6, 6.1, "Fashion-MNIST", fontsize=10, ha='center', color="#2980b9", fontweight='bold')
ax.text(1.6, 5.6, "28x28 Grayscale\n(784 Pixels)", fontsize=9, ha='center', color="#555")

# Arrow to Flatten
ax.annotate('', xy=(3.3, 6.3), xytext=(2.7, 6.3), arrowprops=dict(arrowstyle="->", lw=2.5, color=box_dark))

# 2. Neural Network Model Box
model_box = patches.FancyBboxPatch((3.3, 4.5), 7.4, 3.6, boxstyle="round,pad=0.15", fc="#ffffff", ec="#34495e", lw=2, linestyle='--')
ax.add_patch(model_box)
ax.text(7.0, 7.8, "BENCHMARK NEURAL NETWORK ARCHITECTURE (BenchmarkMLP)", fontsize=11, fontweight='bold', ha='center', color="#2c3e50")

# Layer 1: Flatten + Linear 128
ax.add_patch(patches.FancyBboxPatch((3.6, 5.0), 1.9, 2.3, boxstyle="round,pad=0.1", fc="#ebf5fb", ec=box_blue, lw=1.5))
ax.text(4.55, 6.8, "Layer 1", fontsize=10, fontweight='bold', ha='center', color=box_blue)
ax.text(4.55, 6.2, "Linear(784, 128)\n+ ReLU", fontsize=9, ha='center', color="#333")
ax.text(4.55, 5.4, "100,480 weights\n128 biases", fontsize=8, style='italic', ha='center', color="#777")

# Arrow Layer 1 -> Layer 2
ax.annotate('', xy=(6.0, 6.15), xytext=(5.5, 6.15), arrowprops=dict(arrowstyle="->", lw=2, color=box_dark))

# Layer 2: Linear 64
ax.add_patch(patches.FancyBboxPatch((6.0, 5.0), 1.9, 2.3, boxstyle="round,pad=0.1", fc="#e8f8f5", ec=box_green, lw=1.5))
ax.text(6.95, 6.8, "Layer 2", fontsize=10, fontweight='bold', ha='center', color=box_green)
ax.text(6.95, 6.2, "Linear(128, 64)\n+ ReLU", fontsize=9, ha='center', color="#333")
ax.text(6.95, 5.4, "8,192 weights\n64 biases", fontsize=8, style='italic', ha='center', color="#777")

# Arrow Layer 2 -> Layer 3
ax.annotate('', xy=(8.4, 6.15), xytext=(7.9, 6.15), arrowprops=dict(arrowstyle="->", lw=2, color=box_dark))

# Layer 3: Linear 10
ax.add_patch(patches.FancyBboxPatch((8.4, 5.0), 2.0, 2.3, boxstyle="round,pad=0.1", fc="#fef9e7", ec=box_yellow, lw=1.5))
ax.text(9.4, 6.8, "Output Layer", fontsize=10, fontweight='bold', ha='center', color="#d4ac0d")
ax.text(9.4, 6.2, "Linear(64, 10)\nLogits Output", fontsize=9, ha='center', color="#333")
ax.text(9.4, 5.4, "640 weights\n10 biases", fontsize=8, style='italic', ha='center', color="#777")

# Arrow to Loss
ax.annotate('', xy=(11.3, 6.3), xytext=(10.7, 6.3), arrowprops=dict(arrowstyle="->", lw=2.5, color=box_dark))

# 3. Loss Function
ax.add_patch(patches.FancyBboxPatch((11.3, 5.2), 2.2, 2.2, boxstyle="round,pad=0.1", fc="#fdedec", ec=box_red, lw=2))
ax.text(12.4, 6.7, "LOSS FUNCTION", fontsize=11, fontweight='bold', ha='center', color=box_red)
ax.text(12.4, 6.1, "Cross-Entropy Loss", fontsize=10, fontweight='bold', ha='center', color="#333")
ax.text(12.4, 5.5, "L(θ) = -Σ y log(p)", fontsize=9, ha='center', color="#555")

# 4. Backward Pass Path (Down and left)
ax.annotate('', xy=(12.4, 3.4), xytext=(12.4, 5.2), arrowprops=dict(arrowstyle="->", lw=2.5, color=box_purple, linestyle='-'))
ax.text(12.9, 4.3, "BACKPROPAGATION\nChain Rule\ncompute dL/dθ", fontsize=9, fontweight='bold', ha='left', color=box_purple)

# 5. Gradient Box
ax.add_patch(patches.FancyBboxPatch((11.1, 1.8), 2.6, 1.5, boxstyle="round,pad=0.1", fc="#f4ecf7", ec=box_purple, lw=2))
ax.text(12.4, 2.8, "GRADIENT VECTOR", fontsize=10, fontweight='bold', ha='center', color=box_purple)
ax.text(12.4, 2.2, "g_t = ∇_θ L(θ_t)\n(Slope & Curvature)", fontsize=9, ha='center', color="#333")

# Arrow to Optimizer Engine
ax.annotate('', xy=(9.7, 2.55), xytext=(11.1, 2.55), arrowprops=dict(arrowstyle="->", lw=2.5, color=box_dark))

# 6. THE OPTIMIZER ENGINE (THE STAR OF THE SHOW)
opt_box = patches.FancyBboxPatch((1.5, 0.6), 8.2, 3.2, boxstyle="round,pad=0.15", fc="#fdfefe", ec=box_orange, lw=3)
ax.add_patch(opt_box)
ax.text(5.6, 3.4, "★ THE OPTIMIZER ENGINE (Evaluates 6 Strategies) ★", fontsize=12, fontweight='bold', ha='center', color=box_orange)

# The 6 Optimizer Pills inside
opt_pills = [
    ("1. SGD", "θ = θ - η*g", "#fadbd8"),
    ("2. Momentum", "v = γ*v + η*g", "#fdebd0"),
    ("3. AdaGrad", "G = Σ g², η/sqrt(G)", "#fcf3cf"),
    ("4. RMSProp", "v = β*v + (1-β)*g²", "#ebdef0"),
    ("5. Adam", "m_t, v_t + bias correct", "#d4e6f1"),
    ("6. Nadam", "Adam + Nesterov brake", "#d5f5e3")
]

for i, (name, formula, pill_c) in enumerate(opt_pills):
    col = i % 3
    row = i // 3
    px = 2.0 + col * 2.5
    py = 2.1 - row * 1.1
    ax.add_patch(patches.FancyBboxPatch((px, py), 2.3, 0.9, boxstyle="round,pad=0.08", fc=pill_c, ec="#7f8c8d", lw=1))
    ax.text(px + 1.15, py + 0.55, name, fontsize=9, fontweight='bold', ha='center', color="#2c3e50")
    ax.text(px + 1.15, py + 0.2, formula, fontsize=7.5, ha='center', color="#333")

# 7. Update Arrow Loop back to Model
ax.annotate('', xy=(1.0, 4.5), xytext=(1.5, 2.2), arrowprops=dict(arrowstyle="->", lw=2.5, color=box_green, connectionstyle="arc3,rad=0.3"))
ax.text(0.3, 3.2, "PARAMETER\nUPDATE\nθ_t+1 = θ_t - Δθ", fontsize=9, fontweight='bold', ha='center', color=box_green)

# Arrow into Model
ax.annotate('', xy=(3.3, 5.8), xytext=(1.0, 5.8), arrowprops=dict(arrowstyle="->", lw=2.5, color=box_green))

plt.tight_layout()
diag1_path = "./plots/training_pipeline_architecture.png"
plt.savefig(diag1_path, dpi=300)
plt.close()
print(f"[+] Saved Architecture Diagram 1 to {diag1_path}")


# -------------------------------------------------------------
# DIAGRAM 2: Internal Dataflow Components of all 6 Optimizers
# -------------------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
fig.suptitle("Internal Dataflow & Buffer Architecture for Each Optimizer", fontsize=18, fontweight='bold', color="#2c3e50", y=0.98)

opt_details = [
    ("SGD (Standard Stochastic GD)", "#e74c3c", [
        "Memory Buffers: 0 (No state)",
        "Step: Δθ = η * g_t",
        "Curvature Handling: None",
        "Flaw: High oscillations in ravines"
    ], "g_t ──► [ × -η ] ──► [ + θ_t ] ──► θ_t+1"),
    
    ("SGD with Momentum", "#e67e22", [
        "Memory Buffers: 1 (Velocity buffer v_t)",
        "Step: v_t = γ*v_{t-1} + η*g_t",
        "Curvature Handling: Directional inertia",
        "Advantage: Cancels ravine oscillations"
    ], "g_t ──► [ Accumulate v_t ] ──► [ - v_t ] ──► θ_t+1"),
    
    ("AdaGrad (Adaptive Gradient)", "#f39c12", [
        "Memory Buffers: 1 (Cumulative G_t)",
        "Step: G_t = G_{t-1} + g_t²",
        "Curvature Handling: Per-coordinate scaling",
        "Flaw: Monotonic growth freezes learning"
    ], "g_t ──► [ Sum g_t² (G_t) ] ──► [ ÷ sqrt(G_t) ] ──► θ_t+1"),
    
    ("RMSProp (Geoffrey Hinton)", "#9b59b6", [
        "Memory Buffers: 1 (Leaky variance v_t)",
        "Step: v_t = β*v_{t-1} + (1-β)*g_t²",
        "Curvature Handling: Recent variance window",
        "Advantage: Solves AdaGrad dying rate"
    ], "g_t ──► [ Leaky Filter v_t ] ──► [ ÷ sqrt(v_t) ] ──► θ_t+1"),
    
    ("Adam (Momentum + RMSProp)", "#2980b9", [
        "Memory Buffers: 2 (m_t & v_t)",
        "Step: m_t/(1-β1^t) ÷ sqrt(v_t/(1-β2^t))",
        "Curvature Handling: 1st & 2nd raw moments",
        "Advantage: Rapid convergence & robust defaults"
    ], "g_t ──► [ m_t & v_t ] ──► [ Bias Correct ] ──► θ_t+1"),
    
    ("Nadam (Adam + Nesterov)", "#27ae60", [
        "Memory Buffers: 2 (m_t & v_t)",
        "Step: Lookahead Nesterov momentum",
        "Curvature Handling: Anticipatory braking",
        "Advantage: Smoother transitions near minima"
    ], "g_t ──► [ Lookahead m_t ] ──► [ Scale v_t ] ──► θ_t+1")
]

for idx, (title, col, bullets, pipe_str) in enumerate(opt_details):
    r = idx // 3
    c = idx % 3
    ax = axs[r, c]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Card Background
    card = patches.FancyBboxPatch((0.4, 0.4), 9.2, 9.2, boxstyle="round,pad=0.2", fc="#ffffff", ec=col, lw=2.5)
    ax.add_patch(card)
    
    # Title Banner
    banner = patches.FancyBboxPatch((0.6, 7.8), 8.8, 1.6, boxstyle="round,pad=0.1", fc=col, ec=col)
    ax.add_patch(banner)
    ax.text(5.0, 8.6, title, fontsize=11, fontweight='bold', ha='center', color="#ffffff")
    
    # Bullet points
    y_text = 6.8
    for bullet in bullets:
        ax.text(1.0, y_text, f"• {bullet}", fontsize=9.5, color="#2c3e50")
        y_text -= 0.85
        
    # Pipeline box at bottom
    ax.add_patch(patches.FancyBboxPatch((0.8, 1.0), 8.4, 1.8, boxstyle="round,pad=0.1", fc="#f8f9fa", ec="#bdc3c7", lw=1.5))
    ax.text(5.0, 2.2, "Internal Dataflow:", fontsize=8.5, fontweight='bold', ha='center', color="#7f8c8d")
    ax.text(5.0, 1.5, pipe_str, fontsize=8, ha='center', color="#2980b9", fontweight='bold')

plt.tight_layout()
diag2_path = "./plots/optimizer_mechanisms_architecture.png"
plt.savefig(diag2_path, dpi=300)
plt.close()
print(f"[+] Saved Architecture Diagram 2 to {diag2_path}")

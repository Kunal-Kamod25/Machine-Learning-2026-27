# Optimizers Presentation & Benchmark Project
> **Topic:** Deep Dive into Optimizers — SGD, Momentum, AdaGrad, RMSProp, Adam & Nadam  
> **Target:** 10–15 Minute Team Presentation & Empirical Benchmark  
> **Duration:** 3 Days  

---

## 👥 Team Roles & Responsibilities

| Member | Assigned Topics | Core Theme / Story |
| :--- | :--- | :--- |
| **Person 1** | • Foundations (Loss surface, Gradients, Learning Rate)<br>• **Standard SGD**<br>• **SGD with Momentum** | *"How do we move downhill, and how do we gain speed without getting stuck or zig-zagging?"* |
| **Person 2** | • **AdaGrad**<br>• **RMSProp** | *"Why should every parameter have the same step size? Enter adaptive learning rates and exponential decay."* |
| **Person 3** | • **Adam**<br>• **Nadam**<br>• The Generalization Debate (Adam vs. SGD+M) | *"Combining momentum with adaptive learning, adding lookahead, and why faster isn't always better."* |
| **All 3 Together** | • Common Experiment benchmark<br>• Reading & explaining curves & comparison table<br>• Slide rehearsals & Viva Q&A | *"One common benchmark, fair comparison, and cross-examination defense."* |

---

## 🗓️ 3-Day Execution Timeline

* **Day 1: Theory Mastery & Visual Intuition**
  * Understand the foundational concepts together.
  * Master the 12-point breakdown for all 6 optimizers.
  * Understand *why* each optimizer was created to fix the previous one.
* **Day 2: The Hands-on Experiment & Result Analysis**
  * Run the benchmark script on a standard dataset (e.g. Fashion-MNIST).
  * Generate training/validation loss curves, accuracy curves, and learning rate sensitivity plots.
  * Analyze actual outputs and complete the comparison table.
* **Day 3: Slide Deck Assembly, Scripts & Viva Mock Drill**
  * Build the 10–12 slide deck with clear diagrams and equations.
  * Rehearse the 12–15 minute presentation with speaking scripts for each person.
  * Practice viva and teacher follow-up questions.

---

## 📋 Project Roadmap & Checklist

### Phase 1: Foundational Theory & 12-Point Optimizer Breakdown
- [ ] **Step 1.1**: Foundations (Loss surface, gradient vector, learning rate, weight update rule).
- [ ] **Step 1.2 (Person 1)**: SGD & SGD + Momentum (The blind hiker & heavy ball rolling down a valley).
- [ ] **Step 1.3 (Person 2)**: AdaGrad & RMSProp (Frequent vs. rare features, vanishing learning rate, leaky average).
- [ ] **Step 1.4 (Person 3)**: Adam & Nadam (The gold standard combination, bias correction, Nesterov lookahead).
- [ ] **Step 1.5 (All)**: The Generalization Debate (Why Adam can converge faster but SGD + Momentum may yield better generalization).

### Phase 2: The Practical Experiment (Code & Results)
- [ ] **Step 2.1**: Set up identical benchmark conditions (Fashion-MNIST, identical MLP/CNN architecture, fixed seed, identical batch size & epochs).
- [ ] **Step 2.2**: Implement the Python script comparing SGD, SGD+M, AdaGrad, RMSProp, Adam, and Nadam.
- [ ] **Step 2.3**: Train and save performance plots (Training Loss, Validation Loss, Accuracy vs. Epochs).
- [ ] **Step 2.4**: Run learning-rate sensitivity test ($10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}$).
- [ ] **Step 2.5**: Construct the final comparison table (Convergence speed, Peak validation accuracy, Stability, Tuning effort).

### Phase 3: Teacher Scenarios & Viva Defense
- [ ] **Step 3.1**: Scenario reasoning (Sparse gradients, noisy gradients, saddle points, high/low learning rate).
- [ ] **Step 3.2**: Prepare 4-tier answers for each question:
  1. Beginner analogy
  2. Technical explanation
  3. Underlying mathematical reason
  4. Crisp 10-second Viva punchline

### Phase 4: Presentation Architecture & Speaking Scripts
- [ ] **Step 4.1**: Slide-by-slide structure (10–12 slides max for 10–15 minutes).
- [ ] **Step 4.2**: Exact speaking script for Person 1 (Minutes 0–4).
- [ ] **Step 4.3**: Exact speaking script for Person 2 (Minutes 4–8).
- [ ] **Step 4.4**: Exact speaking script for Person 3 (Minutes 8–12).
- [ ] **Step 4.5**: Joint conclusion & demo walkthrough (Minutes 12–14).
- [ ] **Step 4.6**: Anticipated teacher grilling questions & prepared responses.

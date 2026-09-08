# 10–15 Minute Team Presentation Guide & Word-for-Word Scripts

> **Team Presenters:**  
> • **Kunal (Presenter 1):** Foundations, Standard SGD, and SGD with Momentum  
> • **Rahul (Presenter 2):** Adaptive Learning Rates, AdaGrad, and RMSProp  
> • **Pankaj (Presenter 3):** Adam, Nadam, The Generalization Mystery, and Benchmark Results  
> • **All Members:** Benchmark Walkthrough, Architecture Pipelines, Manual Calculations & Viva Defense  

---

## ⏱️ Presentation Timing & Flow (Total: ~12–14 Minutes)

* **Kunal (0:00 – 4:00):** Welcome, Foundations, Loss Landscapes, Standard SGD, and Momentum.
* **Rahul (4:00 – 7:45):** Why Adaptive Rates are needed, AdaGrad, Dying Learning Rate, and RMSProp.
* **Pankaj (7:45 – 11:30):** The Adam Unification, Nadam Lookahead, and the Flat vs. Sharp Minima Generalization Mystery.
* **All Together (11:30 – 14:00):** Pipeline Architectures, Manual Hand-Calculation Demo, Empirical 1,000-Sample Benchmark & Conclusion.
* **Q&A / Viva Defense (~3–5 minutes):** Teacher grilling.

---

## 🖥️ Slide-by-Slide Deck Outline (12 Slides)

### Slide 1: Title Slide (Shared)
* **Title:** Deep Dive into Deep Learning Optimizers: From SGD to Adam & Nadam
* **Subtitle:** Mathematical Mechanics, Internal Pipelines, Empirical Benchmarks & Generalization Trade-offs
* **Presenters:** Kunal, Rahul, Pankaj
* **Script:**
  > *(Kunal)*: *"Good morning respected faculty and classmates. Today, my teammates Rahul, Pankaj, and I will take you inside the mathematical engine of neural network training: Optimizers."*

---

### Slide 2: The Core Problem — Navigating the Loss Surface (Kunal)
* **Key Visual:** 3D Loss Landscape with fog, valleys, saddle points, and a compass.
* **Key Equations:**
  * Loss function: $L(\theta)$
  * Gradient: $g_t = \nabla_\theta L(\theta_t)$
  * Golden update rule: $\theta_{t+1} = \theta_t - \eta \cdot g_t$
* **Speaker Script (Kunal):**
  > *"When we train a neural network, our goal is to find the parameter weights $\theta$ that minimize the loss function $L(\theta)$. Mathematically, this is an optimization problem over a high-dimensional, non-convex landscape filled with steep ravines, plateaus, and saddle points.  
  > The gradient vector is our compass—it always points in the direction of steepest uphill ascent. By moving in the opposite direction scaled by our step size, the learning rate $\eta$, we take steps downhill.  
  > But how big should each step be? Should all parameters move at the same speed? What happens when gradients oscillate? That is the job of the optimizer. Today, we will explain the internal pipelines, the exact mathematics, and the empirical trade-offs of the six core optimizers."*

---

### Slide 3: Stochastic Gradient Descent (SGD) — The Fast Explorer (Kunal)
* **Key Visual:** Comparison diagram of Full Batch GD vs. Mini-Batch SGD, showing oscillations in a ravine.
* **Key Equation:** $\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L_B(\theta_t)$
* **Internal Pipeline:** Purely reactive, zero internal memory buffers.
* **Speaker Script (Kunal):**
  > *"We begin with Stochastic Gradient Descent (SGD). Full-batch gradient descent evaluates the entire dataset before making a single update—which is computationally impossible on large datasets.  
  > SGD solves this by computing the gradient on small mini-batches. While this makes updates orders of magnitude faster and introduces healthy stochastic noise to escape shallow local minima, it suffers from a fatal flaw: ravines.  
  > In an elongated valley where one wall is much steeper than the floor, SGD oscillates violently from side to side across the walls, making painfully slow progress along the valley base. To solve this, researchers turned to physics: Momentum."*

---

### Slide 4: SGD with Momentum — Adding Physical Inertia (Kunal)
* **Key Visual:** Heavy bowling ball rolling down a valley, canceling lateral jitter and accelerating downhill.
* **Internal Architecture:** Adds a Velocity Buffer state $v_t$.
* **Key Equations:**
  $$v_{t+1} = \gamma v_t + \eta g_t$$
  $$\theta_{t+1} = \theta_t - v_{t+1}$$
* **Speaker Script (Kunal):**
  > *"SGD with Momentum introduces physical inertia. Imagine a heavy bowling ball rolling down a valley. Instead of letting only the current slope dictate the step, Momentum accumulates a running velocity vector $v_t$ that remembers past directions with friction coefficient $\gamma$, typically 0.9.  
  > In ravines, the oscillating gradients have alternating signs and cancel each other out, while gradients pointing down the valley add up constructively, accelerating the ball forward. Momentum also helps the model roll straight through flat saddle points.  
  > However, both SGD and Momentum share one fundamental limitation: they use the exact same global learning rate for every single parameter. In real-world data, some features are frequent while others are rare. I now invite Rahul to explain adaptive learning rates."*

---

### Slide 5: AdaGrad — Parameter-Specific Step Sizes (Rahul)
* **Key Visual:** Two workers: one frequent (small careful steps), one rare (large bold steps).
* **Internal Architecture:** Cumulative squared gradient accumulator $G_t = \sum g^2$.
* **Key Equations:**
  $$G_t = G_{t-1} + g_t^2$$
  $$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$$
* **Speaker Script (Rahul):**
  > *"Thank you, Kunal. In real-world applications, especially in natural language processing or sparse recommendation data, frequent features receive constant gradient updates, while rare, highly informative features receive gradients only once in a while. A single global learning rate cannot serve both.  
  > In 2011, AdaGrad solved this by introducing an adaptive per-parameter learning rate. It maintains an accumulator $G_t$ that sums the squares of all past gradients for every parameter. It then divides the learning rate by $\sqrt{G_t + \epsilon}$.  
  > If a parameter experiences frequent gradients, $G_t$ grows large, shrinking its effective step size to prevent overshooting. If a feature is rare, $G_t$ remains tiny, giving it large, bold steps.  
  > But AdaGrad had a fatal disease: because squared gradients are strictly positive, $G_t$ increases monotonically with every step. Eventually, the denominator becomes so huge that the learning rate drops to practically zero, freezing the model permanently before reaching the minimum."*

---

### Slide 6: RMSProp — The Leaky Memory Cure (Rahul)
* **Key Visual:** Elephant memory (all history) vs. Human sliding-window memory (recent history).
* **Internal Architecture:** Leaky exponential moving average of squared gradients ($v_t$).
* **Key Equations:**
  $$v_t = \beta v_{t-1} + (1 - \beta) g_t^2$$
  $$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t + \epsilon}} \odot g_t$$
* **Speaker Script (Rahul):**
  > *"In 2012, Geoffrey Hinton proposed RMSProp to cure AdaGrad's premature freezing.  
  > Instead of accumulating gradients from the beginning of time, RMSProp replaces the sum with an exponentially decaying moving average with discount factor $\beta$, typically 0.9.  
  > This restricts the memory to a recent sliding window of steps. If terrain was steep 1,000 steps ago but is now flat, RMSProp forgets ancient history and allows the step size to expand again. Learning never freezes.  
  > RMSProp became the gold standard for Recurrent Neural Networks. But notice: RMSProp solved the adaptive step size problem, while Momentum solved the directional velocity problem. What if we combined both? I now hand over to Pankaj to explain Adam."*

---

### Slide 7: Adam — The Grand Unification (Pankaj)
* **Key Visual:** Fusion diagram: Adam = 1st Moment (Momentum $m_t$) + 2nd Moment (RMSProp $v_t$) + Bias Corrections.
* **Key Equations:**
  $$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \quad \hat{m}_t = \frac{m_t}{1-\beta_1^t}$$
  $$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
  $$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
* **Speaker Script (Pankaj):**
  > *"Thank you, Rahul. In 2014, Kingma and Ba introduced Adam—Adaptive Moment Estimation. Adam unites the best of both worlds.  
  > It computes two running moments: the first moment $m_t$ provides directional velocity like a rolling ball, and the second moment $v_t$ scales step sizes coordinate-by-coordinate like RMSProp.  
  > Crucially, Adam introduces mathematical bias corrections: dividing by $(1-\beta_1^t)$ and $(1-\beta_2^t)$. Because moment vectors are initialized at zero, early estimates would be biased toward zero and severely sluggish without this correction.  
  > Adam works out-of-the-box across nearly every deep learning architecture with default hyperparameters, making it the most widely used optimizer in modern AI."*

---

### Slide 8: Nadam & The Generalization Mystery (Pankaj)
* **Key Visual:**
  * Left: Nesterov lookahead anticipatory braking.
  * Right: Sharp Minimum (Adam overfits) vs. Flat Minimum (SGD+M generalizes).
* **Key Equations:**
  * Nadam lookahead: $\bar{m}_t = \beta_1 \hat{m}_t + \frac{1-\beta_1}{1-\beta_1^t} g_t$
* **Speaker Script (Pankaj):**
  > *"Building on Adam, Nadam incorporates Nesterov Accelerated Gradient. Instead of taking a step and then evaluating where you are, Nadam looks ahead at where momentum is carrying you and applies an anticipatory brake, smoothing out updates near sharp minima.  
  > But this leads us to one of the most critical questions in machine learning:  
  > If Adam converges so fast, why do leading researchers still train state-of-the-art vision models using SGD with Momentum?  
  > The answer lies in the geometry of the loss landscape:  
  > Adam rapidly shrinks learning rates along steep dimensions, causing it to fall into sharp, narrow minima. In a sharp minimum, training error is low, but when tested on unseen data, a slight distributional shift causes a massive error spike!  
  > SGD with Momentum, due to its heavy uniform inertia, bounces out of sharp crevices and settles into broad, flat minima. In a flat minimum, the model generalizes significantly better to unseen data. Fast optimization does not always mean a superior final model!"*

---

### Slide 9: Optimizer Architecture Pipeline & Manual Hand Calculations (All 3)
* **Key Visual:**
  * Pipeline flowchart: Forward $\to$ Loss $\to$ Backprop $\to$ Optimizer Engine $\to$ Weights.
  * Side-by-side manual calculation table showing Step 1 and Step 2 arithmetic for a toy weight $w_0 = 1.0, \eta=0.1$.
* **Speaker Script (Kunal & Rahul):**
  > *(Kunal)*: *"To prove that we understand the internal mechanics beyond abstract code, here is the universal optimizer pipeline and an exact manual calculation. With initial weight $1.0$ and learning rate $0.1$, in step 2 with gradient $1.0$, pure SGD takes a step of $0.1$, while Momentum—due to accumulated velocity—takes a step of $0.28$, accelerating progress."*  
  > *(Rahul)*: *"And on the adaptive side, AdaGrad's denominator grew from $4.0 \to 5.0$, immediately shrinking its step size from $0.1$ to $0.044$, whereas RMSProp's leaky memory decayed past gradients, keeping updates healthy. In Adam, dividing by $(1 - 0.999^1)$ scaled $v_1$ from $0.004$ back to $4.0$, completely eliminating initialization bias."*

---

### Slide 10: Our Empirical Benchmark (1,000 Samples on Fashion-MNIST) (All 3)
* **Key Visual:** Experiment architecture: 1,000 images $\to$ MLP ($784 \to 128 \to 64 \to 10$) with cloned initial weights $\theta_0$.
* **Speaker Script (Kunal):**
  > *"To validate these mathematical properties, we designed a completely fair empirical experiment. We trained an identical 3-layer MLP on 1,000 image samples of Fashion-MNIST starting from the exact same cloned weight checkpoint $\theta_0$.  
  > Using 1,000 samples allows each optimizer to complete 10 epochs in under 8.5 seconds, providing a 100% reproducible live demo on CPU."*

---

### Slide 11: Benchmark Results & Sensitivity Analysis (All 3)
* **Key Visual:** 4-panel comparison curves + Learning-Rate Sensitivity plot ($10^{-1}$ to $10^{-4}$).
* **Empirical Data:**
  * All 6 optimizers finished in $7.4$s to $8.7$s.
  * AdaGrad reached 79.8% validation accuracy in early epochs.
  * Adam & Nadam achieved peak validation accuracy (80.1% and 80.3%).
  * At $\eta = 10^{-4}$, pure SGD collapsed to 7.0% (random guessing), while AdaGrad and Adam maintained strong accuracy due to adaptive scaling.
* **Speaker Script (Pankaj):**
  > *"As seen in our live curves, Adam and Nadam achieve the lowest training loss and highest peak accuracy. In our learning-rate sensitivity analysis across four orders of magnitude, standard SGD collapsed completely when the learning rate was small, whereas Adam remained remarkably stable from $10^{-2}$ to $10^{-3}$, and AdaGrad showed flat stability due to its adaptive denominator."*

---

### Slide 12: Conclusion & Master Decision Framework (All 3)
* **Key Visual:** The Decision Flowchart:
  * Sparse Embeddings / NLP Word2Vec $\to$ **AdaGrad / Adam**
  * RNNs / Non-Stationary Reinforcement Learning $\to$ **RMSProp**
  * Transformers & Large Language Models $\to$ **Adam / Nadam**
  * State-of-the-Art Computer Vision $\to$ **SGD + Momentum (with Cosine Decay)**
* **Speaker Script (All):**
  > *(Kunal)*: *"In conclusion, no single optimizer is universally best for every domain."*  
  > *(Rahul)*: *"Optimization is an engineering trade-off between speed, memory overhead, and parameter sensitivity."*  
  > *(Pankaj)*: *"Thank you, and Kunal, Rahul, and I are now eager to take your questions."*

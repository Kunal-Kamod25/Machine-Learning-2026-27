# Project TODO & Progress Tracker

## Phase 1: Foundational Theory & 12-Point Optimizer Breakdown
- [x] **Step 1.1**: Foundational Story (Loss surface, gradient vector, learning rate, weight update rule) -> *See [docs/01_theory_and_optimizers.md](./docs/01_theory_and_optimizers.md)*
- [x] **Step 1.2 (Kunal)**: SGD & SGD + Momentum (The blind hiker & heavy ball rolling down a valley) -> *See [docs/01_theory_and_optimizers.md](./docs/01_theory_and_optimizers.md)*
- [x] **Step 1.3 (Rahul)**: AdaGrad & RMSProp (Frequent vs. rare features, vanishing learning rate, leaky average) -> *See [docs/01_theory_and_optimizers.md](./docs/01_theory_and_optimizers.md)*
- [x] **Step 1.4 (Pankaj)**: Adam & Nadam (The gold standard combination, bias correction, Nesterov lookahead) -> *See [docs/01_theory_and_optimizers.md](./docs/01_theory_and_optimizers.md)*
- [x] **Step 1.5 (All)**: Generalization gap (Why Adam can converge faster but SGD + Momentum may yield better generalization) -> *See [docs/01_theory_and_optimizers.md](./docs/01_theory_and_optimizers.md)*

## Phase 2: The Practical Experiment (Code & Results)
- [x] **Step 2.1**: Set up identical benchmark conditions (Fashion-MNIST, identical MLP architecture, fixed seed, identical initial weights) -> *See [experiment.py](./experiment.py)*
- [x] **Step 2.2**: Implement the Python script comparing SGD, SGD+M, AdaGrad, RMSProp, Adam, and Nadam -> *See [experiment.py](./experiment.py)*
- [x] **Step 2.3**: Train and save performance plots (Training Loss, Validation Loss, Accuracy vs. Epochs) -> *Saved in [plots/all_optimizers_comparison.png](./plots/all_optimizers_comparison.png)*
- [x] **Step 2.4**: Run learning-rate sensitivity test ($10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}$) -> *Saved in [plots/learning_rate_sensitivity.png](./plots/learning_rate_sensitivity.png)*
- [x] **Step 2.5**: Construct the final comparison table (Convergence speed, Peak validation accuracy, Stability, Tuning effort) -> *Saved in [plots/benchmark_summary.csv](./plots/benchmark_summary.csv)*
- [x] **Step 2.6**: Create interactive, fully pre-executed Jupyter Notebook -> *See [optimizers_benchmark.ipynb](./optimizers_benchmark.ipynb)*

## Phase 3: Teacher Scenarios & Viva Defense
- [x] **Step 3.1**: Scenario reasoning (Sparse gradients, noisy gradients, saddle points, high/low learning rate) -> *See [docs/03_application_scenarios_and_viva.md](./docs/03_application_scenarios_and_viva.md)*
- [x] **Step 3.2**: Prepare 4-tier answers for each question (Beginner analogy, Technical explanation, Mathematical reason, 10-second Viva punchline) -> *See [docs/03_application_scenarios_and_viva.md](./docs/03_application_scenarios_and_viva.md)*
- [x] **Step 3.3**: Real-world NLP & Large Language Model (Transformer) case study -> *See [docs/05_architecture_pipelines_and_manual_walkthrough.md](./docs/05_architecture_pipelines_and_manual_walkthrough.md)*

## Phase 4: Presentation Architecture, Speaking Scripts & Manual Walkthrough
- [x] **Step 4.1**: Slide-by-slide structure (12 slides max for 10–15 minutes) -> *See [docs/04_presentation_slides_and_scripts.md](./docs/04_presentation_slides_and_scripts.md)*
- [x] **Step 4.2**: Exact speaking script for Kunal (Minutes 0–4) -> *See [docs/04_presentation_slides_and_scripts.md](./docs/04_presentation_slides_and_scripts.md)*
- [x] **Step 4.3**: Exact speaking script for Rahul (Minutes 4–8) -> *See [docs/04_presentation_slides_and_scripts.md](./docs/04_presentation_slides_and_scripts.md)*
- [x] **Step 4.4**: Exact speaking script for Pankaj (Minutes 8–12) -> *See [docs/04_presentation_slides_and_scripts.md](./docs/04_presentation_slides_and_scripts.md)*
- [x] **Step 4.5**: Joint conclusion & demo walkthrough (Minutes 12–14) -> *See [docs/04_presentation_slides_and_scripts.md](./docs/04_presentation_slides_and_scripts.md)*
- [x] **Step 4.6**: Anticipated teacher grilling questions & prepared responses -> *See [docs/04_presentation_slides_and_scripts.md](./docs/04_presentation_slides_and_scripts.md)*
- [x] **Step 4.7**: Internal optimizer pipelines & hand-calculation numerical walkthrough -> *See [docs/05_architecture_pipelines_and_manual_walkthrough.md](./docs/05_architecture_pipelines_and_manual_walkthrough.md)*

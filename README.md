# Visualizing Neural Network Decision Processes in Unity – An Interactive Approach to Explainable AI

> Bachelor Thesis Project by **Sven Geiger**
> Hochschule Furtwangen University – Faculty of Business Information Systems
> Supervisor: **Prof. Dr. Simon Albrecht**
> *(Winter Semester 2025/26)*

---

## Project Overview

This Bachelor thesis explores how **Explainable AI (XAI)** concepts can be translated into **interactive, game-based visualizations** using the **Unity Engine**.
The project investigates how neural network decision processes can be made more **intuitive**, **visual**, and **engaging**, especially for non-technical audiences.

Instead of relying on static heatmaps or tables, this system turns a neural network into a **2.5D explorable world**, where users can observe and interact with the network’s internal logic — neuron activations, weight adjustments, and learning dynamics — through visual metaphors and gamified challenges.

---

## Research Question

> **How can the internal processes of a neural network be visualized in an interactive and intuitive way to improve human understanding and trust?**

---

## Objectives

1. **Visualize** the flow of data through a trained neural network in Unity.
2. **Design interactive interfaces** that show how neurons and layers contribute to decisions.
3. **Implement gamified elements** (missions, scoring, exploration) to enhance engagement.
4. **Evaluate** user comprehension and perceived faithfulness through a small user study.

---

## System Architecture

**Python (Model Layer)**

* Train small neural networks (e.g., 2→3→1 MLP or CNN on MNIST/CIFAR).
* Export weights, biases, and activations as `.json`.

**Unity (Visualization Layer)**

* Visualizes neural networks in 2D/2.5D environments.
* Each *scene* represents a learning concept:

  * Scene 1: Backpropagation Explorer
  * Scene 2: Optimizers Playground
  * Scene 3: Activation Functions Explorer
  * Scene 4: Loss & Thresholds
  * Scene 5: Capacity Regularization
  * Scene 6: Attribution Slaiency
  * Scene 7: Counterfactuals (work in progress)
  * Scene 8: Data Geometry (work in progress)
* Shared UI framework with adjustable sliders, toggles, and missions.

**Evaluation Layer**

* Includes a simple *faithfulness metric* (perturbation correlation).
* Optional *user study* with 3–5 participants and Likert feedback.

---

## Gameplay & Features

* **Interactive Visualization:** Explore decision boundaries and model behavior in real time.
* **Gamified Tasks:** Missions guide the user to adjust weights, activations, and thresholds.
* **Dynamic UI:** Unified layout across all scenes for scalability.
* **Reproducibility Tools:**

  * `faithfulness_results.csv`
  * `study_feedback.csv`
  * `README_RUNBOOK.md` (run instructions)
  * Deterministic seeds & dataset imports

---

## Methodological Framework

| Stage                             | Description                                                             |
| --------------------------------- | ----------------------------------------------------------------------- |
| **Design Science Research (DSR)** | Iterative design–build–evaluate approach.                               |
| **Implementation**                | Unity 6 (2D/URP) + C# scripts for visualization logic.                  |
| **Data Source**                   | Synthetic datasets (`blobs`, `moons`, `rings`) and small neural models. |
| **Evaluation**                    | User comprehension, perceived faithfulness, reproducibility.            |

---

## Faithfulness & Evaluation

* **Faithfulness metric:** Perturbation Sensitivity Correlation (F-score ≈ 0.9 expected).
* **User study:** 3–5 participants explore 2 scenes, providing feedback on learning effect.
* **Reproducibility:** All runs documented via JSON seeds, event logs, and static screenshots.

---

## 📁 Repository Structure(wok in progress)

```
XAI-Unity/
│
├── Assets/
│   ├── Scenes/                  # Unity scenes (Backprop, Optimizers, etc.)
│   ├── Scripts/                 # Core C# scripts
│   ├── UI/                      # Shared UI components & sprites
│   ├── Datasets/                # JSON datasets for visualization
│   └── Media/                   # Screenshots and thesis figures
│
├── data/
│   ├── model_data.json          # Exported model weights & activations
│   ├── faithfulness_results.csv
│   └── study_feedback.csv
│
├── docs/
│   ├── README_RUNBOOK.md        # Execution guide & reproducibility steps
│   ├── Design_Guidelines.md     # Final XAI gamification guidelines
│   └── Thesis_Abstract.pdf
│
└── LICENSE
```

---

## Run Instructions

1. **Clone Repository**

   ```bash
   git clone https://github.com/<your-username>/XAI-Unity.git
   ```
2. **Open in Unity 6 (URP Template)**
3. **Load Scene**: `Assets/Scenes/MainMenu.unity`
4. **Press Play** – Explore neural network behavior interactively!

Optional:

* Import your own model JSON under `/data/model_data.json`
* Adjust dataset parameters in `GameManager.cs`

---

## Academic Context

This project fulfills the requirements for the **Bachelor of Science in International Business & Information Systems** at HFU.
It combines **Explainable Artificial Intelligence (XAI)** and **Gamification** under a **Design Science Research** framework.

**Supervisor:** Prof. Dr. Simon Albrecht
**Institution:** Hochschule Furtwangen University
**Semester:** Winter 2025/26

---

## References (Selection)

* Samek, W., Montavon, G., Lapuschkin, S., Anders, C. J., & Müller, K.-R. (2017). *Explainable AI: Interpreting, explaining and visualizing deep learning models*. Springer.
* Hohman, F., Kahng, M., Pienta, R., & Chau, D. H. (2019). *Visual analytics in deep learning: An interrogative survey for the next frontiers.* IEEE TVCG, 25(8), 2674–2693.
* Olah, C., et al. (2018). *The building blocks of interpretability.* Distill.
* Chandrasekaran, S., et al. (2022). *User-centric evaluation of interactive explanations.* JMLR.

---



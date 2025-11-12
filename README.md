# Visualizing Neural Network Decision Processes in Unity – An Interactive Approach to Explainable AI

> Bachelor Thesis Project by **Sven Geiger**
> Hochschule Furtwangen University – Faculty of Business Information Systems
> Supervisor: **Prof. Dr. Simon Albrecht**
> *(Winter Semester 2025/26)*

---

## Overview

This repository contains the practical artifact for the Bachelor thesis
**“Visualizing Neural Network Decision Processes in Unity – An Interactive Approach to Explainable AI.”**

The project investigates how **Explainable AI (XAI)** concepts can be communicated through **interactive visualization** and **gamified exploration** using the **Unity Engine**.
It translates abstract neural-network logic—weights, activations, losses, and optimizers—into an engaging, hands-on learning experience for non-experts.

---

## Research Question

> **How can Unity-based interactive visualization and gamification make neural-network decision processes more transparent and understandable for non-expert users?**

---

## Key Objectives

1. **Visualize** neural-network computations (2 → 3 → 1 MLP) as dynamic decision fields.
2. **Integrate gamification** through missions, feedback, and progression loops.
3. **Quantify faithfulness** between visualization and real model behavior using a perturbation-based F-score.
4. **Evaluate usability** and educational impact in a small exploratory user study.

---

## Architecture

**Model Layer (Python)**

* Train small MLPs on synthetic datasets (`blobs`, `moons`, `rings`).
* Export weights, biases, and activations as JSON.

**Visualization Layer (Unity 6 LTS)**

* 2D/URP interactive scenes built in C#.
* Unified UI (Canvas + Mission Panel + Metric Panel).
* Six core scenes + two experimental extensions:

| Scene | Concept                       | Focus                                    |
| ----- | ----------------------------- | ---------------------------------------- |
| S1    | Backpropagation Explorer      | Visualize gradient flow & weight updates |
| S2    | Optimizers Playground         | Compare SGD, Momentum, Adam trajectories |
| S3    | Activation Functions Explorer | Show non-linear decision boundaries      |
| S4    | Loss & Thresholds             | Tune loss functions and decision cutoffs |
| S5    | Capacity & Regularization     | Visualize overfitting vs generalization  |
| S6    | Attribution & Saliency        | Input feature importance                 |
| S7    | Counterfactual Exploration    | Prototype (what-if analysis)             |
| S8    | Data Geometry & Complexity    | Prototype (dataset manifolds)            |

**Evaluation Layer**

* `EvalLogger` – records accuracy, loss, faithfulness (F).
* `EventLogger` – captures interactions and objectives.
* `CrossSceneComparison` – aggregates results for reproducibility.

---

## Evaluation Results

| Metric                 | Mean Value                | Comment                                                |
| ---------------------- | ------------------------- | ------------------------------------------------------ |
| Faithfulness (F-score) | ≈ 0.89 ± 0.02             | Strong correlation between visual and computed outputs |
| User Study (n = 5)     | Educational Value 4.8 / 5 | High clarity and motivation ratings                    |
| Scenes Tested          | S1 & S3                   | Both achieved stable runtime and positive feedback     |

Participants described the experience as *“learning through play”* and confirmed that visual feedback improved conceptual understanding of neural-network behavior.

---

## Features

* 🎮 **Interactive Visualization** – Real-time updates of decision boundaries and activations.
* 🧩 **Gamified Missions** – Level-based objectives with progress tracking and feedback.
* 📊 **Quantitative Faithfulness** – Perturbation-sensitivity correlation (F-metric).
* 🔁 **Reproducibility Suite** – Deterministic seeds, CSV exports, and session logs.
* 🧠 **Educational Focus** – Bridges machine-learning concepts and interactive learning.

---

## How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/XAI-Unity.git
   ```
2. **Open in Unity 6 LTS (URP template).**
3. Load `Assets/Scenes/MainMenu.unity` and press **Play**.
4. Use sliders and dropdowns to adjust weights, activations, or thresholds.
5. Export results via the **Export button** to `/data/faithfulness_results.csv`.

Optional:

* Replace `model_data.json` with your own trained model.
* Modify dataset parameters in `GameManager.cs`.

---

## Methodological Framework

| Stage                             | Description                                                                         |
| --------------------------------- | ----------------------------------------------------------------------------------- |
| **Design Science Research (DSR)** | Iterative cycle of problem definition, artifact design, evaluation, and reflection. |
| **Quantitative Evaluation**       | Faithfulness F-score via perturbation sensitivity tests.                            |
| **Qualitative Evaluation**        | User study for usability and perceived learning effect.                             |
| **Reproducibility**               | Open code, deterministic datasets, GitHub repository documentation.                 |

---

## Academic Context

This project fulfills the requirements for the
**Bachelor of Science in International Business & Information Systems (IBS7)**
at **Hochschule Furtwangen University**.

It contributes to **Explainable AI (XAI)** research by combining
**visualization, gamification, and faithfulness metrics** within a
**Design Science Research methodology**.

---

## Selected References

* Doshi-Velez, F., & Kim, B. (2017). *Towards a rigorous science of interpretable machine learning.* arXiv:1702.08608.
* Hohman, F., Park, H., Robinson, C., & Stasko, J. (2019). *Visual analytics in deep learning.* IEEE TVCG, 25(8), 2674–2693.
* Sailer, M., & Homner, L. (2020). *The gamification of learning: A meta-analysis on effectiveness.* Educational Psychology Review.
* Olah, C., et al. (2018). *The building blocks of interpretability.* Distill.
* Barredo Arrieta, A., et al. (2020). *Explainable AI: Concepts, taxonomies, and challenges toward responsible AI.* Information Fusion.

---

## License

This project is released under the **MIT License**.
All Unity assets, scripts, and data files are provided for academic and educational use.



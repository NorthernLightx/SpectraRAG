# GraphRAG kill-spike — side-by-side

{"chunks": 250, "bib_flagged_pct": 21.2, "zero_extraction_pct": 0.8, "entities_per_chunk": 4.29, "relations_per_chunk": 3.14, "nodes": 787, "edges": 672, "isolates_pct": 5.8, "communities": 240, "singleton_community_pct": 19.2, "reports_emitted": 240}

## What problem domains do these papers address, and what do they share?
**GraphRAG-global:** These papers address the problem domains of scaling law fitting, task-based AI models, and the relationship between data size, model size, and compute resources. They share the common purpose of supporting research into mental health conditions (schizophrenia and bipolar disorder) and investigating the impact of hyperparameters like batch size and learning rate on model training.

**BM25-RAG:** These papers address the problem domains of robotics, specifically human motion understanding and prediction, and reinforcement learning for robotic manipulation. They share the common goal of improving the ability of robots to understand and predict human actions, particularly in complex, real-world environments. Specifically, they focus on leveraging learned representations of motion – like “Action Motifs” – to capture temporal compositions and semantics, leading to more accurate motion prediction and behavior modeling.

---
## Which methods or techniques are introduced or compared across these papers?
**GraphRAG-global:** SMPL, MotionBERT, USDRL, A4Mer, and the Action Motif Dataset (AMD) are introduced or compared across these papers.

**BM25-RAG:** The following methods and techniques are introduced or compared across these papers:

*   **JEPA:** A method that learns through latent token prediction without hand-crafted augmentations.
*   **A4Mer:** A method that masks latent tokens to consolidate motion segments and capture segment semantics.
*   **Action Motifs:** Semantically meaningful and reusable movement patterns.
*   **DINOv3:** A state-of-the-art vision foundation model used for extracting latent tokens.
*   **Latent-to-Action Policy Optimization (LAPO):** A novel RL algorithm that jointly optimizes the latent reasoning process and action generation.
*   **Value Head with MLP:** Used to estimate state values for RL-based post-training.
*   **Chain-of-Thought (CoT) Reasoning:** Integrating latent CoT reasoning over physical dynamics prior to action execution.
*   **VLA-RL, RL4VLA, SimpleVLA-RL, and TGRPO:** RL methods utilizing PPO or GRPO for generalization in unseen environments.
*   **Contrastive Learning:** Methods that require carefully designed augmentations.
*   **Diffusion-based and Flow-matching-based Architectures:** Different VLA architectures.
*   **Incorporating 3D spatial information or point-cloud reasoning:** Enhancements to the VLA backbone.
*   **Embedded CoT:** A method to strengthen spatial reasoning.

---
## What evaluation metrics or benchmarks recur across these papers?
**GraphRAG-global:** Mean R2, σ2, µm(x), t-SNE visualization, BIC, scaling-law instances, ∆Vintra, and ∆Vinter.

**BM25-RAG:** The papers utilize HiK, AMD, and functional manipulation benchmark (FMB) as evaluation metrics and benchmarks. They also compare methods using k-NN classification and a head trained on HiK.

---
## What limitations or open problems do these papers acknowledge?
**GraphRAG-global:** The papers acknowledge “misspecification” when using lr&bsz in the lr&bsz Scaling Laws and Data Analysis cluster.

**BM25-RAG:** The proposed method depends on a mixture-based approximation to multimodal parameter uncertainty, which may be inaccurate when local optima are poorly identified or when the scaling law is severely misspecified. Moreover, the acquisition rule is one-step and does not explicitly optimize long-horizon budget allocation.

---
## What datasets are used across these papers?
**GraphRAG-global:** The following datasets are mentioned across the papers:

*   Action Motif Dataset (AMD)
*   RGB(+D) videos
*   MoCap systems
*   MPJPE
*   H2OT
*   All Data
*   SMPL model

**BM25-RAG:** Based on the context, the following datasets are used:

*   BridgeV2
*   Nyu Franka Play
*   Kuka
*   Stanford Hydra
*   Fractal
*   RoboMIND
*   Robo-Net
*   Jaco Play
*   Language Table
*   Dobb-E
*   BC-Z
*   Toto
*   Maniskill
*   Furniture Bench
*   DROID
*   Utokyo Pr2 Tabletop
*   Roboset
*   Utokyo Xarm Pap
*   FMB Dataset
*   CMU Stretch
*   Taco Play
*   DLR Sara Grid Clamp
*   RoboTurk
*   Utokyo Pr2 Fridge
*   Humans in Kitchens (HiK)

---
## What are the main contributions claimed across these papers?
**GraphRAG-global:** Based on the provided context, the main contributions are:

*   **BIC Optimization for Mixture Models:** A method for optimizing the BIC within a mixture model framework.
*   **V+inter Calculation via Basin Mixture Model:** Utilizing a basin mixture model to calculate V+inter(x, y).
*   **A4Mer:** Learning hierarchical representations of human motion using 3D pose sequences.
*   **Task Class Definitions and Action Association:** A system for classifying actions into distinct task classes.
*   **VLA Models:** Improving VLA models through reinforcement learning (RL).
*   **Budget-Efficient Scaling Law Fitting:** A research focus on budget-efficient scaling law fitting using Active Experiment Selection.

**BM25-RAG:** Based on the provided context, the main contributions are:

1.  **LaST-R1:** A unified VLA framework integrating latent CoT reasoning and RL post-training.
2.  **LAPO:** A novel RL algorithm jointly optimizing latent reasoning and action generation.
3.  **Adaptive Latent CoT Mechanism:** Dynamically adjusts the reasoning horizon based on task diversity.

---
## How do these papers relate to scaling, efficiency, or optimization?
**GraphRAG-global:** These papers relate to scaling, efficiency, or optimization through: budget-efficient scaling law fitting, scaling laws governing compute and model performance, lr&bsz scaling laws and data analysis, optimizing batch size and learning rate, and an automated run selection methodology.

**BM25-RAG:** The papers relate to scaling, efficiency, and optimization through the study of a scaling-law fitting problem with a budget-constrained sequential setting. Specifically, they investigate how to prioritize experiments based on their expected predictive benefit relative to their cost, and how to optimize reasoning lengths and episode lengths to achieve efficient post-training without complex reward modeling.

---
## What future-work directions are suggested across these papers?
**GraphRAG-global:** Based on the provided context, here's a breakdown of suggested future work directions:

*   **VLA Models:** Preparing VLA models for real-world evaluation and out-of-distribution visual scenarios through reinforcement learning.
*   **Budget-Efficient Scaling Law Fitting:** Continued development and application of Active Experiment Selection for scaling law fitting.
*   **Scaling Laws Govern Compute and Model Performance:** Utilizing scaling laws within experimental design and applying uncertainty-aware methods for fitting the scaling state.
*   **lr&bsz Scaling Laws and Data Analysis:** Further investigation into the relationship between learning rate and batch size, potentially addressing observed “misspecification” with the parametric law.

**BM25-RAG:** The papers suggest several future work directions: more robust posterior approximations, multi-step budget-aware design, extensions to broader scaling settings with richer experiment spaces and more realistic cost models; adaptive latent CoT mechanism that dynamically adjusts the reasoning horizon based on task complexity; and integrating physical latent reasoning into VLA learning and jointly optimizing reasoning and action through RL.

---
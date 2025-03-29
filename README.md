

# DemandClean

**DemandClean: A Multi-Objective Learning Framework for Balancing Model Tolerance to Data Authenticity and Diversity**

This repository contains the implementation for our paper:  
**"DemandClean: A Multi-Objective Learning Framework for Balancing Model Tolerance to Data Authenticity and Diversity"**, which proposes a reinforcement learning-based framework to explore how machine learning models tolerate and respond to various types of data quality issues.

---

## 🔍 Overview

DemandClean introduces a unified simulation environment that injects multiple types of data errors (missing values, outliers, and noise) and trains a **DQN-based RL agent** to make optimal cleaning decisions (`no-op`, `repair`, or `delete`) on a per-cell basis.

It balances two competing goals:
- **Authenticity**: preserving real-world data semantics by avoiding unnecessary modifications.
- **Diversity**: maintaining enough data variation and completeness for model generalization.

Our system evaluates and visualizes:
- Cleaning strategies under varying error rates;
- Model tolerance boundaries for different error types;
- Strategy shifts between repairing and deleting based on feature importance and error severity.

---

## 📁 Project Structure

DemandClean/
├── Data/                  # Datasets (adult, Bank, beers, Sick, etc.)
├── saved_models/          # Trained DQN agents
│   └── default_agent.h5
├── DQN_extract.py         # RL inference utilities
├── experiments.py         # Main experiment pipeline
├── README.md              # This file

---

## ⚙️ Installation

You can install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```
If requirements.txt is not provided, install manually:

pip install numpy pandas scikit-learn tensorflow gym tqdm matplotlib

We recommend Python 3.9+.


---
🚀 How to Run

To run the full experiment pipeline:
```bash
python experiments.py
```
By default, this runs the classification task using an SVM downstream model. You can modify parameters in main():
```
task_type = 'classification'  # or 'regression'
n_episodes = 50               # training episodes per error rate
error_rates = [0.1, 0.2, ...] # error severity levels
model_type = 'svm'            # downstream model: 'random_forest', 'logistic_regression', etc.
```
During training, a DQN agent learns to clean data in an environment simulating dirty features and evaluates performance using multiple strategies.

---

📊 Visualization & Results

The framework automatically generates and saves:
	•	Bar charts comparing strategies (Do Nothing, Delete All, Repair All, DemandClean);
	•	Line plots showing action trends vs. error rate;
	•	Tolerance boundary analysis:
	•	Overall model tolerance to errors
	•	Preference shift from repair to delete
	•	Sensitivity to missing vs. outlier errors
	•	Final results saved in experiment_results.json

Sample generated figures:
	•	strategy_comparison*.png
	•	tolerance_threshold*.png
	•	repair_vs_delete_threshold*.png

⸻

🧠 Core Features
	•	✅ Custom data generation for classification/regression
	•	✅ Error injectors (missing, outliers, noise)
	•	✅ Reinforcement learning-based cleaning agent (DQN)
	•	✅ Gym-style cleaning environment
	•	✅ Unified evaluation of multiple cleaning strategies
	•	✅ Tolerance boundary detection and visualization

⸻

📜 Citation

If you find this project useful, please consider citing our paper.

[//]: # (@article{your2025demandclean,)

[//]: # (  title={DemandClean: A Multi-Objective Learning Framework for Balancing Model Tolerance to Data Authenticity and Diversity},)

[//]: # (  author={Your Name and Coauthors},)

[//]: # (  journal={Preprint / Conference},)

[//]: # (  year={2025})

[//]: # (})



[//]: # (⸻)

[//]: # ()
[//]: # (🤝 Acknowledgements)

[//]: # ()
[//]: # (This project is developed and maintained by [Your Name] and collaborators at [Your Lab/Institution].)

[//]: # ()
[//]: # (For questions, contributions, or feedback, please open an issue or contact us directly.)

[//]: # ()
[//]: # (⸻)
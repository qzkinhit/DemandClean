# DemandClean: A Multi-Objective Learning Framework for Balancing Model Tolerance to Data Authenticity and Diversity

This repository contains the implementation for our paper:  
**"DemandClean: A Multi-Objective Learning Framework for Balancing Model Tolerance to Data Authenticity and Diversity"**, which proposes a reinforcement learning-based framework to explore how machine learning models tolerate and respond to various types of data quality issues.

---

## 🔍 Overview

DemandClean introduces a unified simulation environment that injects multiple types of data errors (Missing Errors, Semantic Errors, and Syntactic Errors) and trains a **DQN-based RL agent** to make optimal cleaning decisions (`no-op`, `repair`, or `delete`) on a per-cell basis.

It balances two competing goals:
- **Authenticity**: preserving real-world data semantics by avoiding unnecessary modifications.
- **Diversity**: maintaining enough data variation and completeness for model generalization.

Our system evaluates and visualizes:
- Cleaning strategies under varying error rates;
- Model tolerance boundaries for different error types;
- Strategy shifts between repairing and deleting based on feature importance and error severity.

---

## 📁 Project Structure
```
DemandClean/
├── Data/                  # Datasets (adult, Bank, beers, Sick, etc.)
├── saved_models/          # Trained DQN agents
│   └── default_agent.h5
├── DQN_extract.py         # RL inference utilities
├── experiments.py         # Main experiment pipeline
├── WelcomeDemandClean.py  # 🌐 Streamlit-based front-end UI
├── requirements.txt       # Python dependencies
├── README.md              # This file
```
---

## ⚙️ Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

If requirements.txt is not available, install manually:

pip install numpy pandas scikit-learn tensorflow gym tqdm matplotlib streamlit

💡 Recommended: Python 3.9+

---

🚀 How to Run

1. Run the Core Experiment Pipeline
```bash
python experiments.py
```
By default, this runs the classification task using an SVM downstream model. You can customize:
```
task_type = 'classification'       # or 'regression'
n_episodes = 50                    # training episodes per error rate
error_rates = [0.1, 0.2, 0.3, ...] # error severity levels
model_type = 'svm'                 # downstream model: 'random_forest', 'logistic_regression', etc.
```
2. Launch the Interactive Streamlit UI 🌐

To try an interactive front-end for exploring DemandClean:
```bash
streamlit run WelcomeDemandClean.py
```
---

## 📊 Visualization & Results

The framework automatically generates and saves:

- 📊 **Bar charts** comparing strategies:  
  *Do Nothing*, *Delete All*, *Repair All*, *DemandClean*

- 📈 **plots** showing action trends vs. error rate

- 📉 **Tolerance boundary analysis**:
  - Overall model tolerance to errors  
  - Strategy preference shift (*repair* vs. *delete*)  
  - Error-type-specific sensitivity (*missing* vs. *outlier*)

- 📁 Final results saved in `experiment_results.json`

---

## 🧠 Core Features

- ✅ **Custom synthetic data generation** (classification/regression)
- ✅ **Error injectors**: Missing Errors, Semantic Errors, and Syntactic Errors
- ✅ **DQN-based reinforcement learning cleaning agent**
- ✅ **OpenAI Gym-style environment** for modeling data cleaning as sequential decision making
- ✅ **Multiple cleaning strategy evaluations**
- ✅ **Tolerance boundary estimation**
- ✅ **Interactive Streamlit front-end** for user-defined data exploration

---

📜 Citation

If you find this project helpful, please consider citing:
```
@article{demo4demandclean,
  title={DemandClean: A Multi-Objective Learning Framework for Balancing Model Tolerance to Data Authenticity and Diversity},
  author={Zekai Qian and Xiaoou Ding and Hongzhi Wang},
  year={2025}
}
```
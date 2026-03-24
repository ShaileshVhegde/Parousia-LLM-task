<div align="center">
  
# 🚢 Titanic Survival Prediction: Advanced ML Pipeline  
  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?logo=pandas)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Data_Visualization-4E9A06?logo=python)](https://seaborn.pydata.org/)

*A modular, production-grade Machine Learning pipeline designed to predict Titanic passenger survival using advanced feature engineering and heavily tuned classification algorithms.*

</div>

---

## 📖 Table of Contents
1. [Overview & Problem Statement](#-overview--problem-statement)
2. [🤖 The Models](#-the-models)
3. [🚀 Pipeline Architecture](#-pipeline-architecture)
4. [📂 Repository Structure](#-repository-structure)
5. [📈 Real-time Visualizations](#-real-time-visualizations)
6. [💻 Installation & Usage](#-installation--usage)

---

## 🎯 Overview & Problem Statement
The sinking of the Titanic is one of the most infamous shipwrecks in history. The challenge? Build a highly accurate predictive model to answer: **"What sorts of people were more likely to survive?"** using deep passenger data (name, age, socio-economic class, cabin geography, etc.). 

This repository skips basic scripts by deploying a **rigid, scalable Sklearn architecture** that prevents data leakage and rigorously tests parameters in an automated environment.

---

## 🤖 The Models

As per project standards, this pipeline explicitly pits a strong linear baseline against an advanced ensemble tree architecture to highlight the impact of non-linear algorithms:

### 1️⃣ Logistic Regression (The Baseline)
- **Purpose**: Serves as our statistical benchmark. Logistic Regression maps the linear relationships between the engineered features (like Class or Gender) directly to the odds of survival.
- **Hyperparameter Tuning**: We dynamically search the regularization parameter `C` within our Scikit-learn pipeline to ensure the model isn't overfitting to the training noise.

### 2️⃣ Gradient Boosting Classifier (The Champion)
- **Purpose**: A highly aggressive, non-linear ensemble algorithm. Gradient Boosting builds sequential decision trees—where every new tree focuses entirely on correcting the errors (residuals) made by the previous trees.
- **Hyperparameter Tuning**: Tuned extensively via 5-Fold Cross Validation (`GridSearchCV`). We actively search across:
  - `n_estimators` (Number of trees: 100, 200, 300)
  - `learning_rate` (Impact of each tree)
  - `max_depth` (Preventing overly massive trees)

---

## 🚀 Pipeline Architecture

This project is built directly entirely around `sklearn.pipeline.Pipeline`, guaranteeing industry-standard execution:

1. **Automated Feature Engineering (`TitanicFeatureEngineer`)**
   - Extracts semantic `Title` clusters from passenger names (Mr., Mrs., Miss., Rare).
   - Combines Siblings and Parents to compute `FamilySize` and an `IsAlone` binary flag.
   - Intelligently chunks `AgeGroup` and `FareBin` into quantiles to strip away continuous noise.
2. **Leak-Free Preprocessing**
   - Automatically imputes median/mode logic strictly inside the training folds.
   - Replaces faulty Label Encoding with mathematically sound `pd.get_dummies(drop_first=True)` One-Hot Encoding via a unified `ColumnTransformer`.
3. **Decoupled Evaluation Matrix**
   - Validates **both models** simultaneously against strictly isolated 20% stratified test sets.

---

## 📂 Repository Structure
```text
.
├── main.py                    # 🚦 Primary execution file orchestrating the entire flow
├── data_preprocessing.py      # ⚙️ ColumnTransformer and Sklearn Pipeline definitions
├── feature_engineering.py     # 🧬 Custom Python class containing Feature transformations
├── model_training.py          # 🧠 Dictionary holding Model instances & GridSearch grids
├── evaluation.py              # 📊 Mathematical evaluator extracting Accuracy, Precision, F1
├── utils.py                   # 🛠️ Reusable helper functions
├── requirements.txt           # 📦 Required dependency packages
└── README.md                  # 📜 This documentation
```

---

## 📈 Real-time Visualizations

Evaluating purely on raw accuracy isn't enough. When you execute this codebase, a unique **`/plots`** folder is generated on your machine containing presentation-ready graphics:
* **Model Comparison**: A dynamic Bar Chart precisely charting Model 1 vs Model 2 accuracies.
* **ROC & AUC Curve comparisons**: Illustrating True Positive vs False Positive thresholds.
* **Confusion Matrices**: Direct heatmaps generated per model exposing False Negatives and False Positives.
* **Feature Importance**: A Gini-importance heatmap revealing the exact generated variables (e.g. `Title_Mr`, `Sex_male`) that dictated Gradient Boosting's survival selections.

---

## 💻 Installation & Usage

### 1. Requirements
Ensure you have Python 3.8+ installed on your workspace. We highly recommend using a virtual environment (`venv`).

### 2. Setup
Clone the repository, verify `titanic.csv` is in your root directory, and install dependencies:
```bash
# Install all required data science libraries natively
pip install -r requirements.txt
```

### 3. Execution
Simply trigger the main orchestrator! The script will automatically trigger hyperparameter searches, evaluate the baseline alongside Gradient Boosting, export the visual graphics, and loudly declare the best model in your terminal.
```bash
python main.py
```

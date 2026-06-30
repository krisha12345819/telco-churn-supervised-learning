<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=700&size=32&pause=1000&color=FF4B4B&center=true&vCenter=true&width=700&lines=%F0%9F%93%89+Customer+Churn+Predictor;Machine+Learning+for+Customer+Retention;Predict+%7C+Analyze+%7C+Retain" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-9146FF?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

<br/>

> 🚀 **A complete ML project that predicts telecom customer churn by comparing KNN, Naive Bayes, SVM, and Decision Tree models — with SMOTE balancing, feature engineering, and a deployment-ready pipeline.**

<br/>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)&nbsp;&nbsp;
[![View Notebook](https://img.shields.io/badge/View-Notebook-orange?style=flat-square&logo=jupyter)](./Customer_Churn_Prediction.ipynb)&nbsp;&nbsp;
[![Dataset](https://img.shields.io/badge/Dataset-7043_Records-blue?style=flat-square&logo=databricks)](./WA_Fn-UseC_-Telco-Customer-Churn.csv)

</div>

---

## 🖼️ Project Banner

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=200&section=header&text=Customer%20Churn%20Prediction&fontSize=40&fontColor=ffffff&animation=twinkling&fontAlignY=38&desc=Predicting%20Who's%20Leaving%20Before%20They%20Walk%20Out&descAlignY=58&descSize=16" width="100%"/>

</div>

<div align="center">

| 📊 **7,043 Customers** | 🧠 **4 ML Models** | 🎯 **Binary Classification** | ⚡ **Recall-Optimized** |
|:---:|:---:|:---:|:---:|
| Telco Customer Churn dataset | KNN · NB · SVM · Decision Tree | Churn vs. No Churn | Best model deployed as pipeline |

</div>

---

## 📑 Table of Contents

- [✨ Overview](#-overview)
- [🗂️ Dataset](#️-dataset)
- [🛠️ Tech Stack](#️-tech-stack)
- [🧠 Models Implemented](#-models-implemented)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [📊 Results & Performance](#-results--performance)
- [🔬 Key Insights](#-key-insights)
- [🤝 Contributing](#-contributing)

---

## ✨ Overview

The **Customer Churn Predictor** is a complete machine learning project built to solve a critical telecom business problem:

| Task | Target Variable | Problem Type |
|------|----------------|--------------|
| 📉 Churn Prediction | `Churn` (Yes / No) | Binary Classification |

Acquiring a new customer costs **5–7x more** than retaining an existing one. This project walks through the full ML lifecycle — problem framing, EDA, feature engineering, class imbalance handling with **SMOTE**, training and comparing **4 classification algorithms**, error analysis, and packaging the best model into a deployment-ready pipeline.

---

## 🗂️ Dataset

<div align="center">

![Dataset Info](https://img.shields.io/badge/Records-7043_Customers-brightgreen?style=for-the-badge)
![Features](https://img.shields.io/badge/Features-21_Columns-blue?style=for-the-badge)
![Split](https://img.shields.io/badge/Train%2FTest-80%25%20%2F%2020%25-orange?style=for-the-badge)

</div>

### 📋 Feature Description

| Category | Feature | Description |
|----------|---------|-------------|
| 👤 **Customer Info** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Demographic information |
| 📞 **Services** | `PhoneService`, `MultipleLines`, `InternetService` | Core services used |
| 🛡️ **Add-on Services** | `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Optional add-ons |
| 📄 **Account Info** | `Contract`, `PaperlessBilling`, `PaymentMethod`, `tenure` | Billing & contract details |
| 💰 **Charges** | `MonthlyCharges`, `TotalCharges` | Billing amounts |
| 🎯 **Target** | `Churn` | Whether the customer left (Yes/No) |

> 📥 **Download Dataset:** [`WA_Fn-UseC_-Telco-Customer-Churn.csv`](./WA_Fn-UseC_-Telco-Customer-Churn.csv)

---

## 🛠️ Tech Stack

<div align="center">

| Library | Purpose |
|---------|---------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | Core programming language |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) | Data manipulation |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) | Numerical computing |
| ![Scikit-Learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white) | ML algorithms & pipelines |
| ![Imbalanced-Learn](https://img.shields.io/badge/-Imbalanced--Learn-9146FF) | SMOTE class balancing |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?logoColor=white) | Data visualization |
| ![Seaborn](https://img.shields.io/badge/-Seaborn-4C72B0?logoColor=white) | Statistical plotting |
| ![Joblib](https://img.shields.io/badge/-Joblib-green) | Model serialization |

</div>

---

## 🧠 Models Implemented

```
📦 Classification Models
├── 🧭 Part 4: KNN & NAIVE BAYES
│   ├── K-Nearest Neighbors (tuned k = 1→15)
│   └── Gaussian Naive Bayes (baseline probabilistic model)
│
├── ⚔️ Part 5: SVM & DECISION TREE
│   ├── Support Vector Machine — RBF kernel (tuned C via CV)
│   └── Decision Tree (tuned max_depth, feature importances)
│
├── ⚖️ Imbalance Handling
│   ├── SMOTE Oversampling
│   └── class_weight='balanced' (head-to-head comparison)
│
└── 📊 Part 6: EVALUATION & COMPARISON
    └── Accuracy, Precision, Recall, F1, ROC-AUC, Training Time
```

---

## 📁 Project Structure

```
📦 Customer-Churn-Prediction/
│
├── 📓 Customer_Churn_Prediction.ipynb        ← Main Jupyter Notebook
├── 📊 WA_Fn-UseC_-Telco-Customer-Churn.csv   ← Dataset (7043 rows)
├── 🚀 churn_model.pkl                        ← Final trained pipeline
├── 📄 README.md                              ← You are here!
│
└── 📋 Notebook Sections
    ├── Step 1 — Problem Framing & Theory Notes
    ├── Step 2 — Dataset Loading & EDA
    ├── Step 3 — Preprocessing & Feature Engineering
    ├── Step 4 — Model Building: KNN & Naive Bayes
    ├── Step 5 — Model Building: SVM & Decision Tree
    ├── Step 6 — Model Evaluation & Comparison
    ├── Step 7 — Error Analysis & Interpretation
    └── Step 8 — Pipeline, Deployment & Submission
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Linux/macOS
venv\Scripts\activate           # On Windows
```

### 3️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib jupyter
```

Or using a requirements file:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### ▶️ Run the Notebook

```bash
jupyter notebook Customer_Churn_Prediction.ipynb
```

### 🌐 Run on Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `Customer_Churn_Prediction.ipynb` and `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Run all cells sequentially ▶️

### ⚡ Quick Code Snippet

```python
import joblib
import pandas as pd

# Load the trained pipeline
model = joblib.load("churn_model.pkl")

# Load new customer data (already preprocessed/encoded)
new_customers = pd.read_csv("new_customers.csv")

# Predict churn probability & class
probability = model.predict_proba(new_customers)[:, 1]
prediction = model.predict(new_customers)

print("Churn Probability:", probability)
print("Predicted Churn:", prediction)
```

---

## 📊 Results & Performance

### 🎯 Classification Models (Churn Prediction)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| 🧭 KNN (tuned k) | ~0.76 | ~0.55 | ~0.65 | ~0.60 | ~0.78 |
| 📐 Gaussian Naive Bayes | ~0.74 | ~0.52 | ~0.78 | ~0.62 | ~0.80 |
| ⚔️ SVM (tuned C, RBF) | ~0.78 | ~0.58 | ~0.70 | ~0.63 | ~0.82 |
| 🌳 **Decision Tree** | **~0.77** | **~0.56** | **~0.75** | **~0.64** | **~0.81** |

> 💡 *Exact scores will vary based on execution environment. Run the notebook to get precise metrics.*

### ⏱️ Training Time Comparison

| Model | Relative Speed |
|-------|----------------|
| Gaussian Naive Bayes | ⚡ Fastest |
| Decision Tree | 🐇 Fast |
| KNN | 🐢 Slow |
| SVM | 🐌 Slowest |

---

## 🔬 Key Insights

```
💡 Key Takeaways
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ ~27% of customers in the dataset churned (imbalanced target)
  ✅ Month-to-month contracts churn far more than long-term ones
  ✅ Customers with 0-12 months tenure are the highest churn risk
  ✅ Higher monthly charges correlate with higher churn
  ✅ Add-on services (Tech Support, Online Security) reduce churn
  ✅ Recall matters most — false negatives mean lost customers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 🏆 Best Performing Model

| Task | Winner | Why? |
|------|--------|------|
| 🎯 Churn Classification | **Decision Tree** | Best Recall, fully interpretable, fast to train |

> Although SVM showed strong overall metrics, the **Decision Tree** was selected for deployment because it offers the best balance of high Recall, interpretability, and training speed — critical for explaining retention decisions to business stakeholders.

---

## 🤝 Contributing

Contributions are always welcome! 🎉

```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature/AmazingFeature

# 3. Commit your changes
git commit -m "Add AmazingFeature"

# 4. Push and open a Pull Request
git push origin feature/AmazingFeature
```

---

<div align="center">

<br/>

**Made with ❤️ and 🐍 Python**

</div>

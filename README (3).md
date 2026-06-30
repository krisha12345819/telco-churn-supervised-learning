<div align="center">

![Banner](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=180&section=header&text=Customer%20Churn%20Prediction&fontSize=38&fontColor=ffffff&animation=fadeIn&fontAlignY=40)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](#)
[![Status](https://img.shields.io/badge/Status-Completed-2EA44F?style=for-the-badge)](#)

</div>

## 🪧 Overview

Acquiring a new customer costs **5–7x more** than keeping an existing one. This project builds a machine learning system that predicts which telecom customers are likely to churn, using the classic **Telco Customer Churn dataset** — 7,043 customers, 21 features.

The notebook walks through the full ML lifecycle: exploratory analysis, feature engineering, imbalance handling, training and comparing four models, error analysis, and packaging the best one into a deployable pipeline.

## 📦 Project Files

- `Customer_Churn_Prediction.ipynb` — the full analysis and modeling notebook
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — raw dataset
- `churn_model.pkl` — final trained pipeline, ready for deployment

## 🛣️ Workflow

1. **Problem Framing** — business value of churn prediction, confusion matrix economics, precision vs. recall trade-offs
2. **Exploratory Data Analysis** — univariate and bivariate analysis, correlation heatmaps
3. **Preprocessing & Feature Engineering** — missing values, encoding, scaling, new features (tenure groups, service counts, autopay)
4. **Handling Class Imbalance** — SMOTE oversampling
5. **Model Training** — KNN, Gaussian Naive Bayes, SVM, Decision Tree
6. **Model Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, training time
7. **Error Analysis** — false negative profiling, feature importance
8. **Deployment** — final pipeline saved as `churn_model.pkl`

## 🔍 Key Insights

- About **27%** of customers in the dataset churned — a moderately imbalanced target
- **Month-to-month contracts** churn far more than one- or two-year contracts
- Customers with **0–12 months tenure** are the highest churn-risk group
- **Higher monthly charges** correlate with higher churn
- Customers with **add-on services** like Tech Support or Online Security churn less
- **Contract type, tenure, and monthly charges** are the top predictive features

## 🤖 Models Trained

| Model | Notes |
|---|---|
| K-Nearest Neighbors | Tuned across `k` values for best F1 score |
| Gaussian Naive Bayes | Fast probabilistic baseline |
| Support Vector Machine (RBF) | `C` tuned via cross-validation |
| Decision Tree | `max_depth` tuned, most interpretable, feature importances extracted |

Recall is the priority metric here — missing a customer who's about to churn (false negative) is far more costly to the business than flagging one who stays (false positive). The notebook also directly compares **SMOTE** vs. **`class_weight='balanced'`** for handling imbalance.

## 🏆 Final Model

The model with the best Recall is wrapped in a `scikit-learn` `Pipeline`, saved with `joblib` as `churn_model.pkl`, and ready to generate predictions on new customer data:

```python
import joblib

model = joblib.load("churn_model.pkl")

probability = model.predict_proba(new_customer_data)[:, 1]   # churn risk score
prediction  = model.predict(new_customer_data)                # 0 = stay, 1 = churn
```

## 🧰 Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · imbalanced-learn (SMOTE) · joblib

## ⚡ Getting Started

```bash
git clone <your-repo-url>
cd customer-churn-prediction

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib jupyter

jupyter notebook Customer_Churn_Prediction.ipynb
```

## 💼 Why It Matters

Identifying high-risk customers before they leave turns a reactive support process into a proactive retention strategy — personalized offers, loyalty perks, or timely outreach, all driven by a single Recall-optimized model.

<div align="center">

⭐ If this project helped you, consider starring the repo!

</div>

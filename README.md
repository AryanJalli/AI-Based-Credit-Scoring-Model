# 🧠 AI-Based Credit Scoring Model
Predicting loan default risk for underbanked populations using Machine Learning models including Logistic Regression, Random Forest, and XGBoost.

## 📌 Project Overview
This project builds an AI-driven credit scoring model using the **Give Me Some Credit** dataset to predict whether a borrower will default on a loan. Traditional credit scoring systems rely heavily on past repayment history, which excludes underbanked individuals. Our goal is to create a fair and data-driven alternative using machine learning and interpretable model insights.

---

## 🗂️ Dataset
**Source:** Give Me Some Credit (Kaggle)  
**Target Variable:** `SeriousDlqin2yrs` (1 = Default, 0 = No Default)  
**Key Features Used:**
- Monthly Income
- Debt Ratio
- Revolving Credit Utilization
- Delinquency History (30/60/90+ days late)
- Number of Open Credit Lines / Real Estate Loans
- Dependents

---

## 🧹 Data Preprocessing
| Task | Technique |
|------|------------|
| Missing values | Median (numeric) & Mode (categorical) imputation |
| Outliers | IQR deletion & capped values for extreme ranges |
| Scaling | StandardScaler for numerical stability |
| Class Imbalance | Weighted models / XGBoost `scale_pos_weight` |
| Feature Engineering | Risk indicators (see below) |

### Engineered Features
| Feature | Description |
|---------|-------------|
| Financial Pressure | `DebtRatio * (Income / Utilization)` |
| Active Credit Burden | `OpenCreditLines + RealEstateLoans` |
| Income Per Dependent | `MonthlyIncome / (Dependents + 1)` |
| High Utilization Flag | Utilization > 0.80 |

---

## 🤖 Models Trained
| Model | Strengths | Weaknesses |
|-------|------------|-------------|
| Logistic Regression | Baseline, interpretable, fast | Struggles with nonlinearity & imbalance |
| Random Forest | Captures interactions, less tuning needed | Medium recall on minority class |
| **XGBoost (Best Model)** | Best ROC-AUC, strong minority class recall, SHAP explainability | Sensitive to hyperparameter tuning |

---

## 📈 Model Performance
**Evaluation Metrics:** ROC-AUC, F1-Score, Precision, Recall

| Model | ROC-AUC | F1-Score | Notes |
|-------|:-------:|:--------:|------|
| Logistic Regression | ~0.82 | 0.17–0.20 | Interpretable but limited |
| Random Forest        | ~0.85 | 0.31–0.33 | Good baseline tree model |
| **XGBoost (Final)**  | **~0.87** | **~0.34** | Best overall performance |

XGBoost was selected as the final model due to its ability to capture nonlinear patterns and class imbalance more effectively than the other approaches.

---

## 🔍 Model Explainability (SHAP)
SHAP values were used to interpret individual predictions and feature importance.

Top predictors influencing default risk:
- Revolving credit utilization
- Past delinquency counts (30/60/90+ days late)
- Active credit burden
- Monthly income pressure

SHAP helped confirm the model was identifying meaningful and financially realistic patterns.

---

## 🚀 How to Run the Project
```bash
# Clone repository
git clone https://github.com/yourusername/credit-scoring-model.git
cd credit-scoring-model

# Install dependencies
pip install -r requirements.txt

# Run training script
python train_model.py

# Credit-Risk-Management-ML-Project

A production-oriented Machine Learning project that predicts loan default risk using classification models and deploys the final model through a Streamlit dashboard.

---

## 📌 Business Objective

Financial institutions face significant losses due to loan defaults.  
This project builds a predictive system that classifies applicants as:

- **0 → Non-Default (Low Risk)**
- **1 → Default (High Risk)**

The goal is to improve loan approval decisions by accurately identifying high-risk borrowers.

---

## 📊 Dataset Summary

- **Rows:** 32,500  
- **Features:** 12  
- **Target Variable:** `loan_status`  
- **Class Distribution:**  
  - 78% Non-Default  
  - 22% Default  

Since the dataset is moderately imbalanced, evaluation metrics beyond accuracy were prioritized.

---

## 🧠 Models Implemented

- Logistic Regression (Baseline)
- Random Forest
- Gradient Boosting
- XGBoost

All models were built using a **Scikit-learn Pipeline** with:

- `ColumnTransformer`
- `StandardScaler`
- `OneHotEncoder`

---

## 📈 Evaluation Strategy

Due to class imbalance, model selection was based on:

- **Recall (Class 1 – Defaulters)**  
- **F1 Score**
- **ROC-AUC**
- Confusion Matrix  

### Why Not Accuracy?

Accuracy can be misleading in imbalanced datasets.  
ROC-AUC and Recall were prioritized to ensure strong detection of defaulters.

---

## 🏆 Final Model

The selected model achieved:

- High ROC-AUC (strong class separation)
- Balanced Precision & Recall
- Improved F1-score

The final trained pipeline was serialized as:

credit_risk_model1.pkl
---

## 🚀 Deployment

A Streamlit dashboard was developed to:

- Display model performance metrics
- Accept real-time user input
- Predict default probability
- Classify applicant risk level

---

## 🛠 Tech Stack

- Python  
- Pandas / NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Joblib  

---
```
## 📂 Project Structure
credit-risk-project/
│
├── EDA_Credit_Risk_Management.ipynb
├── credit_risk_model1.pkl
├── dashboard.py
├── requirements.txt
└── README.md
```

---

## 💼 Key Takeaways

- Applied structured ML workflow using pipelines
- Handled class imbalance using appropriate metrics
- Compared multiple models before final selection
- Built a deployment-ready ML system
- Focused on business-driven model evaluation

---

## 👨‍💻 Author

Anurag Patil  
Machine Learning | Credit Risk Modeling | Data Science

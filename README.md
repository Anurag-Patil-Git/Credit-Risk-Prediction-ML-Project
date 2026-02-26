# 💳 Credit Risk Prediction System

An end-to-end **Machine Learning project** that predicts loan default risk and delivers interactive analytics through a deployed **Streamlit dashboard**.

🔗 **Live App:**  
https://credit-risk-prediction-ml-project.streamlit.app/

---

## 🚀 Project Overview

Financial institutions face significant losses due to loan defaults.  
This project builds a **data-driven credit risk classification system** that helps identify high-risk borrowers before loan approval.

**Prediction Output**
- `0` → Low Risk (Non-Default)
- `1` → High Risk (Default)

The system combines **data analysis, machine learning, and cloud deployment** into a production-ready solution.

---

## 📊 Dataset

- Records: **32,500**
- Features: **12**
- Target: `loan_status`
- Class Distribution:
  - 78% Non-Default
  - 22% Default

Since the dataset is imbalanced, model evaluation focused on **Recall, F1-score, and ROC-AUC** instead of accuracy.

---

## 🧠 Models Compared

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost ✅ (Final Model)

Built using a **Scikit-learn Pipeline** with automated preprocessing:

- ColumnTransformer
- StandardScaler
- OneHotEncoder

---

## 🏆 Final Model

The selected model achieved strong performance in identifying defaulters while maintaining balanced precision and recall.
credit_risk_model1.pkl

---

## 📈 Dashboard Features

✅ Interactive EDA  
✅ Risk Analytics  
✅ Real-time Loan Prediction  
✅ Default Probability Score  
✅ Business KPI Monitoring  

---

## ⚙️ Deployment

The application is deployed using **Streamlit Community Cloud**.

**Workflow**
Model Training → Model Serialization → GitHub → Streamlit Deployment

The deployed app loads the trained pipeline and performs real-time predictions.

---

## 🛠 Tech Stack

**Machine Learning**
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost

**Visualization & App**
- Plotly
- Streamlit

**Deployment**
- GitHub
- Streamlit Cloud
- Joblib

---

## 📂 Project Structure
```
Credit-Risk-Prediction-ML-Project/
│
├── EDA_Credit_Risk_Management.ipynb
├── credit_risk_dataset.csv
├── credit_risk_model1.pkl
├── dashboard.py
├── requirements.txt
└── README.md
```

---

## 💼 Skills Demonstrated

- End-to-End ML Pipeline
- Credit Risk Modeling
- Feature Engineering
- Imbalanced Data Handling
- Model Evaluation
- ML Deployment
- Dashboard Development

---

## 👨‍💻 Author

**Anurag Patil**

🔗 GitHub  
https://github.com/Anurag-Patil-Git  

🔗 LinkedIn  
https://www.linkedin.com/in/anurag-patil/

---

⭐ If you like this project, consider giving it a star!

Saved production model:

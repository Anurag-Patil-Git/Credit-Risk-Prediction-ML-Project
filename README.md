# 💳 Credit Risk Prediction System

An end-to-end **Machine Learning project** that predicts loan default risk using a **FastAPI backend** serving predictions to an interactive **Streamlit frontend/dashboard**.

🔗 **Live App:**  
https://credit-risk-prediction-ml-project.streamlit.app/

---

## 🚀 Project Overview

Financial institutions face significant losses due to loan defaults.  
This project builds a **data-driven credit risk classification system** that helps identify high-risk borrowers before loan approval.

**Prediction Output**
- `0` → Low Risk (Non-Default)
- `1` → High Risk (Default)

**Architecture**: Streamlit (UI) → FastAPI (API/Model Serving) → XGBoost Pipeline

The system combines **data analysis, machine learning, API development, and cloud deployment** into a production-ready full-stack solution.

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

## 📈 Frontend Features (Streamlit)

✅ Interactive EDA  
✅ Risk Analytics  
✅ Real-time Loan Prediction (via FastAPI)  
✅ Default Probability Score  
✅ Business KPI Monitoring  

## 🌐 Backend API Features (FastAPI)

✅ Model serving at `/predict`
✅ Health checks at `/health`
✅ Auto-generated interactive docs at `/docs`
✅ CORS enabled for frontend integration
✅ Production-ready with Uvicorn

---

## ⚙️ Deployment

**Full-Stack Deployment:**

- **Frontend**: Streamlit Community Cloud
- **Backend**: Uvicorn/Gunicorn (Render, Railway, Heroku, etc.)

**Local Workflow** (see QUICKSTART.md):
1. `python main.py` → FastAPI at http://localhost:8000/docs
2. `streamlit run dashboard.py` → UI at http://localhost:8501

**Production Workflow:**
Model Training → Model Serialization → GitHub → Deploy API → Deploy Frontend

---

## 🛠 Tech Stack

**Machine Learning**
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost

**Backend API**
- FastAPI
- Uvicorn
- Pydantic
- Joblib

**Frontend & Visualization**
- Streamlit
- Plotly

**Deployment**
- GitHub
- Streamlit Cloud
- Uvicorn/Gunicorn (API)

---

## 📂 Project Structure
```
Credit-Risk-Prediction-ML-Project/
│
├── EDA_Credit_Risk_Management.ipynb      # Exploratory Data Analysis
├── credit_risk_dataset.csv               # Training dataset
├── credit_risk_model1.pkl                # Trained XGBoost model
├── main.py                              # FastAPI backend (model serving)
├── dashboard.py                         # Streamlit frontend (UI + API client)
├── requirements.txt                      # Dependencies
├── QUICKSTART.md                         # Setup guide
├── README.md                            # This file
└── DEPLOYMENT.md                        # Deployment instructions
```

---

## 💼 Skills Demonstrated

- End-to-End ML Pipeline
- Credit Risk Modeling
- Feature Engineering
- Imbalanced Data Handling
- Model Evaluation
- FastAPI Development
- RESTful API Design
- Full-Stack ML Deployment
- Frontend-Backend Integration
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

# ================================
# FASTAPI BACKEND FOR ML MODEL
# ================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import Optional

# ================= INITIALIZE APP =================
app = FastAPI(
    title="Credit Risk Prediction API",
    description="ML API for predicting credit risk",
    version="1.0.0"
)

# ================= CORS MIDDLEWARE =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= LOAD MODEL =================
@app.on_event("startup")
def load_model():
    global model
    try:
        model_path = os.path.join(
            os.path.dirname(__file__),
            "credit_risk_model1.pkl"
        )
        with open(model_path, "rb") as f:
            model = joblib.load(f)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

# ================= REQUEST MODEL =================
class PredictionRequest(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: int
    loan_amnt: float
    loan_int_rate: float
    loan_intent: str
    loan_grade: str
    cb_person_cred_hist_length: int
    cb_person_default_on_file: str
    loan_percent_income: float
    emp_length_missing: int
    income_stability: float
    dti_band: str

# ================= RESPONSE MODEL =================
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str

# ================= HEALTH CHECK =================
@app.get("/health")
def health_check():
    """Check if API and model are running"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# ================= PREDICTION ENDPOINT =================
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict credit risk for a borrower
    
    Returns:
    - prediction: 0 (Low Risk) or 1 (High Risk)
    - probability: Probability of default (0-1)
    - risk_level: Human-readable risk level
    """
    try:
        # Create DataFrame with proper feature order
        input_data = pd.DataFrame({
            "person_age": [request.person_age],
            "person_income": [request.person_income],
            "person_home_ownership": [request.person_home_ownership],
            "person_emp_length": [request.person_emp_length],
            "loan_amnt": [request.loan_amnt],
            "loan_int_rate": [request.loan_int_rate],
            "loan_intent": [request.loan_intent],
            "loan_grade": [request.loan_grade],
            "cb_person_cred_hist_length": [request.cb_person_cred_hist_length],
            "cb_person_default_on_file": [request.cb_person_default_on_file],
            "loan_percent_income": [request.loan_percent_income],
            "emp_length_missing": [request.emp_length_missing],
            "income_stability": [request.income_stability],
            "dti_band": [request.dti_band]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = "Very High Risk"
        elif probability >= 0.5:
            risk_level = "High Risk"
        elif probability >= 0.3:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=round(probability, 4),
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ================= ROOT ENDPOINT =================
@app.get("/")
def root():
    """API Documentation"""
    return {
        "message": "Credit Risk Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }

# ================= RUN SERVER =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

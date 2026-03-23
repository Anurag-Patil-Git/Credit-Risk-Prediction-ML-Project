# 🚀 Deployment Guide: FastAPI + Streamlit Architecture

This guide explains how to run the refactored Credit Risk Prediction system with a FastAPI backend and Streamlit frontend.

---

## 📋 Architecture Overview

```
┌────────────────────┐         ┌──────────────────┐        ┌──────────────┐
│   Streamlit App    │────────▶│   FastAPI        │───────▶│   ML Model   │
│   (Dashboard)      │ (HTTP)  │   (main.py)      │        │ (pkl file)   │
└────────────────────┘         └──────────────────┘        └──────────────┘
   Port: 8501              Port: 8000
```

---

## 📦 Prerequisites

1. **Python 3.8+** installed
2. All required packages (see Installation below)

---

## 🔧 Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **FastAPI** - Web framework for the ML API
- **Uvicorn** - ASGI server to run FastAPI
- **Streamlit** - Frontend UI
- **Requests** - HTTP client for calling the API
- **scikit-learn, xgboost** - ML dependencies

---

## 🏃 How to Run

### Option 1: Run Both Services (Recommended)

#### Terminal 1 - Start FastAPI Backend

```bash
python main.py
```

Expected output:
```
✅ Model loaded successfully
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### Terminal 2 - Start Streamlit Frontend

```bash
streamlit run dashboard.py
```

Expected output:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Then open **http://localhost:8501** in your browser.

---

### Option 2: Run FastAPI with Auto-Reload (Development)

For development with code changes:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 📚 API Endpoints

The FastAPI server provides the following endpoints:

### 1. Health Check
```bash
GET http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Make Prediction
```bash
POST http://localhost:8000/predict
```

Request body:
```json
{
  "person_age": 30,
  "person_income": 50000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5,
  "loan_amnt": 20000,
  "loan_int_rate": 10.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "cb_person_cred_hist_length": 5,
  "cb_person_default_on_file": "N",
  "loan_percent_income": 0.4,
  "emp_length_missing": 0,
  "income_stability": 8333.33,
  "dti_band": "Medium"
}
```

Response:
```json
{
  "prediction": 0,
  "probability": 0.25,
  "risk_level": "Low Risk"
}
```

### 3. API Documentation
Visit http://localhost:8000/docs for interactive API docs (Swagger UI)

---

## 🔌 Environment Variables (Optional)

You can configure the API URL in the Streamlit app by modifying `dashboard.py`:

```python
API_URL = "http://localhost:8000"  # Change this if API runs on different port/host
```

---

## ✅ Testing

### Test the API Manually (Using curl)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 30,
    "person_income": 50000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_amnt": 20000,
    "loan_int_rate": 10.0,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "cb_person_cred_hist_length": 5,
    "cb_person_default_on_file": "N",
    "loan_percent_income": 0.4,
    "emp_length_missing": 0,
    "income_stability": 8333.33,
    "dti_band": "Medium"
  }'
```

### Test the API Using Python

```python
import requests

payload = {
    "person_age": 30,
    "person_income": 50000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_amnt": 20000,
    "loan_int_rate": 10.0,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "cb_person_cred_hist_length": 5,
    "cb_person_default_on_file": "N",
    "loan_percent_income": 0.4,
    "emp_length_missing": 0,
    "income_stability": 8333.33,
    "dti_band": "Medium"
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
```

---

## 🐛 Troubleshooting

### Issue: "Cannot connect to API"

**Solution:** Make sure the FastAPI server is running in Terminal 1:
```bash
python main.py
```

### Issue: "Model load failed"

**Solution:** Ensure `credit_risk_model1.pkl` exists in the same directory as `main.py`

### Issue: Port 8000 already in use

**Solution:** Run on a different port:
```bash
python main.py --port 9000
```

Then update `dashboard.py`:
```python
API_URL = "http://localhost:9000"
```

### Issue: CORS errors

**Solution:** The API already has CORS enabled. If issues persist, check browser console for details.

---

## 📊 Project Structure

```
Credit-Risk-Prediction-ML-Project/
├── main.py                          # FastAPI Backend
├── dashboard.py                     # Streamlit Frontend
├── credit_risk_model1.pkl           # Trained ML Model
├── credit_risk_dataset.csv          # Dataset
├── requirements.txt                 # Python Dependencies
├── EDA_Creadit_Risk_Management.ipynb # Notebook
├── README.md                        # Project Overview
└── DEPLOYMENT.md                    # This file
```

---

## 🚀 Production Deployment

For production, consider:

1. **API Server (FastAPI):**
   - Use Gunicorn + Uvicorn workers
   - Deploy to Cloud (Azure App Service, AWS EC2, etc.)
   - Set up monitoring and logging

2. **Frontend (Streamlit):**
   - Deploy to Streamlit Cloud
   - Docker containerization
   - Environment variable management

3. **Database:**
   - Store predictions in a database
   - Track audit logs
   - Monitor model performance

---

## 📝 Example: Running with Gunicorn (Production)

```bash
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

---

## 🔐 Security Considerations

1. **API Authentication** - Add API key or JWT tokens
2. **Input Validation** - Already implemented via Pydantic
3. **Rate Limiting** - Consider adding rate limiting middleware
4. **HTTPS** - Use HTTPS in production
5. **Environment Variables** - Don't hardcode API URLs

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review API docs at http://localhost:8000/docs
3. Check FastAPI logs in Terminal 1

---

Happy deploying! 🎉

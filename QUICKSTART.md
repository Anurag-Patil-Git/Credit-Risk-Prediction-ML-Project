# ⚡ Quick Start Guide

## 1️⃣ Install Dependencies (One Time)

```bash
pip install -r requirements.txt
```

## 2️⃣ Start FastAPI Backend (Terminal 1)

```bash
python main.py
```

✅ You should see:
```
✅ Model loaded successfully
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 3️⃣ Start Streamlit App (Terminal 2)

```bash
streamlit run dashboard.py
```

✅ App opens at: **http://localhost:8501**

## 4️⃣ Use the App

1. Go to **Model Prediction** page
2. Fill in borrower details
3. Click **Predict Risk**
4. See the credit risk prediction! 🎯

---

## 🔗 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check if API is running |
| `/predict` | POST | Get risk prediction |
| `/docs` | GET | Interactive API docs |

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI backend (loads model & serves predictions) |
| `dashboard.py` | Streamlit frontend (calls FastAPI) |
| `credit_risk_model1.pkl` | Trained ML model |
| `requirements.txt` | Python dependencies |

---

## 🛠️ Environment Variables (Optional)

If API runs on different port/host, update in `dashboard.py`:

```python
API_URL = "http://your-api-host:your-port"
```

---

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| "Cannot connect to API" | Run `python main.py` first |
| "Model not found" | Ensure `credit_risk_model1.pkl` exists |
| Port 8000 in use | Change port in `main.py` |

---

✨ **You're all set!** The API handles predictions, Streamlit handles the UI. Perfect separation! 🎉

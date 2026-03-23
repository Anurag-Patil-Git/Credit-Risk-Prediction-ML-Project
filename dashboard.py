# ================================
# FIX STREAMLIT FILE WATCHER ISSUE
# ================================
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# ================================
# IMPORTS
# ================================
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Credit Risk Analytics & Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CSS =================
st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background: linear-gradient(180deg,#0f172a,#020617);
    color: white;
}

/* ✅ FIX SIDEBAR VISIBILITY */
[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#1e293b,#0f172a) !important;
}

[data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
}

/* ================= TEXT ================= */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] div,
[data-testid="stAppViewContainer"] label {
    color: white !important;
}

[data-testid="stAppViewContainer"] .stMarkdown,
[data-testid="stAppViewContainer"] .stText {
    color: white !important;
}

/* ================= KPI ================= */
.kpi-card{
    background: linear-gradient(135deg,#2563eb,#1e40af);
    padding:20px;
    border-radius:14px;
    text-align:center;
    color:white;
    box-shadow:0 6px 18px rgba(0,0,0,0.4);
}

.kpi-title{
    font-size:18px;
    opacity:0.9;
}

.kpi-value{
    font-size:34px;
    font-weight:700;
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)


# ================= LOAD DATA =================
@st.cache_data(show_spinner=False)
def load_data():
    csv_url = "https://raw.githubusercontent.com/Anurag-Patil-Git/Credit-Risk-Prediction-ML-Project/main/credit_risk_dataset.csv"
    df = pd.read_csv(csv_url)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


df = load_data()


# ================= API CONFIGURATION =================
API_URL = "http://localhost:8000"

@st.cache_resource
def get_api_health():
    """Check if FastAPI backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

api_healthy = get_api_health()

if not api_healthy:
    st.warning(
        "⚠️ FastAPI backend is not running. Please start the server with:\n"
        "`python main.py`"
    )


# ================= SIDEBAR =================
st.sidebar.title("🏦 Credit Risk Analytics & Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "About Dataset",
        "Univariate Analysis",
        "Bivariate Analysis",
        "Model Prediction"
    ]
)

# ================= KPI SECTION =================
if page != "Model Prediction":

    st.title("💳 Credit Risk Analytics & Prediction")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Customers</div>
            <div class="kpi-value">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Average Income</div>
            <div class="kpi-value">${df['person_income'].mean():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        default_rate = df['loan_status'].mean()*100
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Default Rate</div>
            <div class="kpi-value">{default_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Loan Amount</div>
            <div class="kpi-value">${df['loan_amnt'].mean():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()


# ================= ABOUT DATASET =================

if page == "About Dataset":

    st.header("📘 About the Dataset & Project")

    st.markdown("""
### 🏦 Project Overview

This project analyzes borrower financial profiles to assess **credit risk** and predict the likelihood of loan default.

The dataset contains borrower demographic information, financial details, loan characteristics, and historical credit behavior.  
The objective is to uncover risk patterns and support smarter, data-driven lending decisions.
""")

    st.divider()

    # ---------------- DATASET SUMMARY ---------------- #
    st.subheader("📊 Dataset Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", f"{df.shape[0]:,}")
    col2.metric("Total Features", f"{df.shape[1]}")
    col3.metric("Target Variable", "loan_status")

    st.markdown("""
The dataset is structured for supervised machine learning, where **loan_status** represents whether a borrower defaulted.
""")

    st.divider()

    # ---------------- FEATURE CATEGORIES ---------------- #
    st.subheader("🧾 Feature Categories")

    st.markdown("""
#### 👤 Borrower Demographics
- `person_age` – Age of the borrower  
- `person_income` – Annual income  
- `person_home_ownership` – Housing status (Rent, Own, Mortgage, Other)  
- `person_emp_length` – Employment duration in years  

#### 💳 Loan Details
- `loan_amnt` – Loan amount requested  
- `loan_int_rate` – Interest rate applied  
- `loan_intent` – Purpose of loan  
- `loan_grade` – Credit grade assigned  

#### 📈 Credit History
- `cb_person_cred_hist_length` – Length of credit history  
- `cb_person_default_on_file` – Historical default indicator  

#### 🧠 Engineered Features
- `loan_percent_income` – Loan amount as % of income  
- `emp_length_missing` – Indicator for missing employment length  
- `income_stability` – Stability proxy using income & employment  
- `dti_band` – Debt-to-income risk category  
""")
    

    st.divider()

    # ---------------- TARGET VARIABLE ---------------- #
    st.subheader("🎯 Target Variable: Loan Status")

    st.markdown("""
- `0` → No Default  
- `1` → Default  

This binary classification problem helps financial institutions determine whether a borrower is likely to repay the loan.
""")

    default_rate = df["loan_status"].mean() * 100

    st.info(f"📌 Current Default Rate in Dataset: **{default_rate:.2f}%**")

    st.divider()

    # ---------------- BUSINESS OBJECTIVE ---------------- #
    st.subheader("🎯 Business Objective")

    st.markdown("""
The primary goal of this project is to:

- Reduce loan default risk  
- Improve credit approval strategies  
- Enable risk-based pricing  
- Support automated underwriting systems  
- Enhance portfolio risk monitoring  

By leveraging data analytics and machine learning, banks can minimize financial losses while maintaining responsible lending practices.
""")

    st.divider()

    # ---------------- DATA QUALITY ---------------- #
    st.subheader("🧹 Data Preparation & Quality")

    st.markdown("""
- Removed unnecessary `Unnamed` columns  
- Handled missing employment values  
- Created engineered features for improved prediction  
- Standardized categorical labels  
- Ensured consistency between training and prediction pipelines  
""")

    st.divider()

    # ---------------- PROJECT IMPACT ---------------- #
    st.subheader("🚀 Project Impact")

    st.markdown("""
This dashboard provides:

✔ Interactive exploratory data analysis  
✔ Risk segmentation insights  
✔ Model-based probability prediction  
✔ Executive-level KPI monitoring  

The system bridges **data analysis + machine learning + business intelligence** into a unified fintech solution.
""")


# ================= UNIVARIATE =================
elif page == "Univariate Analysis":

    st.header("📊 Univariate Analysis")

    numeric_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(include='object').columns

    feature_type = st.selectbox(
        "Select Feature Type",
        ["Numeric", "Categorical"]
    )

    if feature_type == "Numeric":

        col = st.selectbox("Choose Feature", numeric_cols)

        fig = px.histogram(
            df,
            x=col,
            nbins=40,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:

        col = st.selectbox("Choose Feature", categorical_cols)

        fig = px.bar(
            df[col].value_counts(),
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)


# ================= BIVARIATE =================
elif page == "Bivariate Analysis":

    st.header("📈 Bivariate Analysis")

    numeric_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(include='object').columns

    chart = st.selectbox(
        "Select Chart",
        ["Scatter", "Box"]
    )

    if chart == "Scatter":

        x = st.selectbox("X Axis", numeric_cols)
        y = st.selectbox("Y Axis", numeric_cols, index=1)

        fig = px.scatter(
            df,
            x=x,
            y=y,
            color="loan_status",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:

        cat = st.selectbox("Categorical", categorical_cols)
        num = st.selectbox("Numeric", numeric_cols)

        fig = px.box(
            df,
            x=cat,
            y=num,
            color=cat,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)


# ================= MODEL PREDICTION =================
elif page == "Model Prediction":

    st.title("Credit Default Prediction")

    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input("Annual Income", 1000, 1000000, 50000)
        age = st.slider("Age", 18, 75, 30)
        emp_length = st.slider("Employment Length", 0, 40, 5)
        credit_hist = st.slider("Credit History Length", 1, 40, 5)
        interest_rate = st.slider("Interest Rate (%)", 5.0, 35.0, 10.0)

    with col2:
        loan_amnt = st.number_input("Loan Amount", 500, 500000, 20000)

        home_ownership = st.selectbox(
            "Home Ownership",
            ["RENT","OWN","MORTGAGE","OTHER"]
        )

        loan_intent = st.selectbox(
            "Loan Intent",
            ["PERSONAL","EDUCATION","MEDICAL",
             "VENTURE","HOMEIMPROVEMENT",
             "DEBTCONSOLIDATION"]
        )

        loan_grade = st.selectbox(
            "Loan Grade",
            ["A","B","C","D","E","F","G"]
        )

        past_default = st.selectbox(
            "Past Default",
            ["Y","N"]
        )

    loan_percent_income = loan_amnt / income
    emp_length_missing = 1 if emp_length == 0 else 0
    income_stability = income / (emp_length + 1)

    dti_band = pd.cut(
        [loan_percent_income],
        bins=[0,0.2,0.4,0.6,1],
        labels=["Low","Medium","High","Very High"]
    )[0]

    predict = st.button("Predict Risk", use_container_width=True)

    if predict:
        if not api_healthy:
            st.error("❌ API is not available. Please start the FastAPI server first.")
        else:
            try:
                # Create payload for API
                payload = {
                    "person_age": age,
                    "person_income": income,
                    "person_home_ownership": home_ownership,
                    "person_emp_length": emp_length,
                    "loan_amnt": loan_amnt,
                    "loan_int_rate": interest_rate,
                    "loan_intent": loan_intent,
                    "loan_grade": loan_grade,
                    "cb_person_cred_hist_length": credit_hist,
                    "cb_person_default_on_file": past_default,
                    "loan_percent_income": loan_percent_income,
                    "emp_length_missing": emp_length_missing,
                    "income_stability": income_stability,
                    "dti_band": str(dti_band)
                }

                # Call FastAPI endpoint
                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=5
                )

                if response.status_code == 200:
                    result = response.json()
                    
                    st.divider()

                    # Display prediction result
                    if result["prediction"] == 1:
                        st.error(f"⚠ {result['risk_level']}")
                    else:
                        st.success(f"✅ {result['risk_level']}")

                    st.metric(
                        "Default Probability",
                        f"{result['probability']*100:.2f}%"
                    )
                else:
                    st.error(f"❌ Prediction failed: {response.text}")
            
            except requests.exceptions.Timeout:
                st.error("❌ API request timeout. Please check if the server is running.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Ensure the FastAPI server is running on http://localhost:8000")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")



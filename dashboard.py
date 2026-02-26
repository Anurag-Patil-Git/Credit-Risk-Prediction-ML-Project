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
import joblib


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
}

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


# ================= LOAD MODEL =================
@st.cache_resource
def load_model():

    model_path = os.path.join(
        os.path.dirname(__file__),
        "credit_risk_model1.pkl"
    )

    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model


model = load_model()


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

    st.header("📘 About Dataset")

    st.write("""
This project analyzes borrower financial profiles to predict loan default risk.
The dashboard provides analytics insights and ML-based prediction.
""")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())


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

        input_data = pd.DataFrame({
            "person_age":[age],
            "person_income":[income],
            "person_home_ownership":[home_ownership],
            "person_emp_length":[emp_length],
            "loan_amnt":[loan_amnt],
            "loan_int_rate":[interest_rate],
            "loan_intent":[loan_intent],
            "loan_grade":[loan_grade],
            "cb_person_cred_hist_length":[credit_hist],
            "cb_person_default_on_file":[past_default],
            "loan_percent_income":[loan_percent_income],
            "emp_length_missing":[emp_length_missing],
            "income_stability":[income_stability],
            "dti_band":[dti_band]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.divider()

        if prediction == 1:
            st.error("⚠ High Risk of Default")
        else:
            st.success("✅ Low Risk Borrower")

        st.metric(
            "Default Probability",
            f"{probability*100:.2f}%"
        )


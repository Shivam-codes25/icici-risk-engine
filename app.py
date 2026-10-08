# app.py
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# This forces the layout to be wide and sets the title
st.set_page_config(page_title="Credit Risk & Underwriting Engine", layout="wide")

# This title should immediately show up on the blank white space!
st.title("🏦 ICICI Bank Underwriting Dashboard")
st.markdown("---")

# --- 1. DATA EXTRACTION & ML TRAINING ---
@st.cache_data
def load_and_train_model():
    conn = sqlite3.connect('icici_risk.db')
    df = pd.read_sql("""
        SELECT a.*, f.operating_income, f.annual_debt_service, f.collateral_value 
        FROM applicants a
        JOIN financials f ON a.applicant_id = f.applicant_id
    """, conn)
    conn.close()
    
    features = ['annual_income', 'existing_debt', 'loan_amount', 'credit_score', 'interest_rate']
    X = df[features]
    y = df['default']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_scaled, y)
    
    return df, model, scaler, features

df, model, scaler, features = load_and_train_model()

# --- 2. SIDEBAR & NAVIGATION ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/12/ICICI_Bank_Logo.svg", width=150)
st.sidebar.title("Underwriting Engine")

input_mode = st.sidebar.radio("Data Source:", ["Manual Entry", "Select from Database"])

if input_mode == "Select from Database":
    applicant_type = "Corporate (Business)" # Synthetic DB is corporate
    applicant_list = df['applicant_id'].tolist()
    selected_app = st.sidebar.selectbox("Select Applicant ID", applicant_list)
    
    app_data = df[df['applicant_id'] == selected_app].iloc[0]
    
    conn = sqlite3.connect('icici_risk.db')
    fin_data = pd.read_sql(f"SELECT * FROM financials WHERE applicant_id='{selected_app}'", conn).iloc[0]
    fx_data = pd.read_sql(f"SELECT * FROM forex WHERE applicant_id='{selected_app}'", conn).iloc[0]
    conn.close()

else:
    selected_app = "Manual Applicant"
    
    applicant_type = st.sidebar.radio("Borrower Type:", ["Corporate (Business)", "Retail (Individual)"])
    st.sidebar.markdown("---")

    st.sidebar.subheader("1. Loan Request Details")
    annual_income = st.sidebar.number_input("Annual Income/Revenue (₹)", min_value=100000, value=1500000, step=100000)
    existing_debt = st.sidebar.number_input("Existing Total Debt (₹)", min_value=0, value=500000, step=50000)
    loan_amount = st.sidebar.number_input("Requested Loan (₹)", min_value=50000, value=2000000, step=100000)
    credit_score = st.sidebar.slider("Credit Score", 300, 900, 750)
    interest_rate = st.sidebar.slider("Interest Rate (%)", 7.0, 18.0, 10.5)
    tenure_months = st.sidebar.selectbox("Tenure (Months)", [12, 24, 36, 48, 60, 72], index=4)

    st.sidebar.subheader(f"2. {'Corporate' if applicant_type == 'Corporate (Business)' else 'Personal'} Financials")
    
    if applicant_type == "Corporate (Business)":
        operating_income = st.sidebar.number_input("Operating Income (₹)", value=800000, step=50000)
        annual_debt_service = st.sidebar.number_input("Annual Debt Service (₹)", value=150000, step=10000)
        collateral_value = st.sidebar.number_input("Collateral Value (₹)", value=2500000, step=100000)
        current_assets = st.sidebar.number_input("Current Assets (₹)", value=1000000)
        current_liabilities = st.sidebar.number_input("Current Liabilities (₹)", value=400000)
        total_debt = st.sidebar.number_input("Total Debt (₹)", value=500000)
        shareholder_equity = st.sidebar.number_input("Shareholder Equity (₹)", value=1200000)
        
        st.sidebar.subheader("3. Forex Exposure")
        usd_receivables = st.sidebar.number_input("USD Receivables ($)", value=50000)
        usd_payables = st.sidebar.number_input("USD Payables ($)", value=20000)
    else:
        operating_income = st.sidebar.number_input("Annual Take-Home Salary (₹)", value=1200000, step=50000)
        annual_debt_service = st.sidebar.number_input("Total Yearly EMIs Paid (₹)", value=150000, step=10000)
        collateral_value = st.sidebar.number_input("Asset Value (Home/Vehicle) (₹)", value=3000000, step=100000)
        current_assets = st.sidebar.number_input("Liquid Assets (Savings/FDs) (₹)", value=500000)
        current_liabilities = st.sidebar.number_input("Short-term Bills/Cards (₹)", value=50000)
        total_debt = st.sidebar.number_input("Total Outstanding Loans (₹)", value=500000)
        shareholder_equity = st.sidebar.number_input("Personal Net Worth (₹)", value=3450000)
        
        usd_receivables = 0
        usd_payables = 0

    app_data = pd.Series({
        'annual_income': annual_income, 'existing_debt': existing_debt, 
        'loan_amount': loan_amount, 'credit_score': credit_score, 
        'interest_rate': interest_rate, 'tenure_months': tenure_months
    })
    fin_data = pd.Series({
        'operating_income': operating_income, 'annual_debt_service': annual_debt_service,
        'collateral_value': collateral_value, 'current_assets': current_assets,
        'current_liabilities': current_liabilities, 'total_debt': total_debt,
        'shareholder_equity': shareholder_equity
    })
    fx_data = pd.Series({
        'usd_receivables': usd_receivables, 'usd_payables': usd_payables
    })

# --- 3. FINANCIAL CALCULATIONS ---
r = (app_data['interest_rate'] / 100) / 12
n = app_data['tenure_months']
p = app_data['loan_amount']
emi = p * (r * (1 + r)**n) / ((1 + r)**n - 1)

monthly_income = app_data['annual_income'] / 12
monthly_debt = (app_data['existing_debt'] / 12) + emi
dti = (monthly_debt / monthly_income) * 100 if monthly_income > 0 else 999

dscr = fin_data['operating_income'] / (fin_data['annual_debt_service'] + (emi*12)) if (fin_data['annual_debt_service'] + (emi*12)) > 0 else 0
ltv = (app_data['loan_amount'] / fin_data['collateral_value']) * 100 if fin_data['collateral_value'] > 0 else 999

# --- 4. ML PREDICTION ---
app_features = app_data[features].values.reshape(1, -1)
app_scaled = scaler.transform(app_features)
pd_score = model.predict_proba(app_scaled)[0][1] * 100

def get_risk_grade(pd):
    if pd < 10: return "A (Low Risk)"
    elif pd < 20: return "B (Moderate Risk)"
    elif pd < 35: return "C (High Risk)"
    else: return "D (Very High Risk)"
    
risk_grade = get_risk_grade(pd_score)

# --- 5. UI VIEWS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Applicant Overview", "Credit Risk", "Financial Health", "Forex Stress Test", "Final Decision"
])

with tab1:
    st.header(f"Applicant Overview: {selected_app}")
    st.subheader(f"Borrower Class: {applicant_type}")
    col1, col2 = st.columns(2)
    col1.metric("Annual Income", f"₹ {app_data['annual_income']:,.2f}")
    col1.metric("Existing Debt", f"₹ {app_data['existing_debt']:,.2f}")
    col1.metric("Credit Score", int(app_data['credit_score']))
    
    col2.metric("Requested Loan", f"₹ {app_data['loan_amount']:,.2f}")
    col2.metric("Tenure", f"{int(app_data['tenure_months'])} Months")
    col2.metric("Interest Rate", f"{app_data['interest_rate']}%")

with tab2:
    st.header("Credit Risk & Ratios")
    col1, col2, col3 = st.columns(3)
    col1.metric("Debt-to-Income (DTI)", f"{dti:.1f}%", "Max target: 40%", delta_color="inverse")
    col2.metric("Est. Monthly EMI", f"₹ {emi:,.2f}")
    col3.metric("Loan-to-Value (LTV)", f"{ltv:.1f}%", "Max target: 80%", delta_color="inverse")
    
    st.subheader("Machine Learning PD Model")
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pd_score,
        title = {'text': "Probability of Default (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 10], 'color': "lightgreen"},
                {'range': [10, 35], 'color': "yellow"},
                {'range': [35, 100], 'color': "red"}]
        }))
    st.plotly_chart(fig)

with tab3:
    if applicant_type == "Corporate (Business)":
        st.header("Corporate Financial Health")
        col3_label = "Debt to Equity"
    else:
        st.header("Personal Financial Health")
        col3_label = "Debt to Net Worth"
        
    current_ratio = fin_data['current_assets'] / fin_data['current_liabilities'] if fin_data['current_liabilities'] > 0 else 0
    debt_to_equity = fin_data['total_debt'] / fin_data['shareholder_equity'] if fin_data['shareholder_equity'] > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Debt Service Coverage (DSCR)", f"{dscr:.2f}x", "Min target: 1.25x")
    col2.metric("Liquidity Ratio", f"{current_ratio:.2f}x")
    col3.metric(col3_label, f"{debt_to_equity:.2f}x")

with tab4:
    if applicant_type == "Retail (Individual)":
        st.info("ℹ️ Forex Stress Testing is typically not applicable for standard retail borrowers. Risk is evaluated entirely on domestic income, credit history, and personal leverage.")
    else:
        st.header("Forex Exposure & Stress Testing")
        base_rate = 83.00
        net_exposure_usd = fx_data['usd_receivables'] - fx_data['usd_payables']
        
        st.write(f"**Net USD Exposure:** ${net_exposure_usd:,.2f}")
        
        scenarios = {
            "Base (₹83)": base_rate,
            "Stress +5% (₹87.15)": base_rate * 1.05,
            "Stress +10% (₹91.30)": base_rate * 1.10
        }
        
        for name, rate in scenarios.items():
            inr_value = net_exposure_usd * rate
            st.metric(name, f"₹ {inr_value:,.2f}", 
                      f"₹ {(inr_value - (net_exposure_usd*base_rate)):,.2f} impact" if name != "Base (₹83)" else None)

with tab5:
    st.header("Underwriting Decision Matrix")
    
    rules_passed = 0
    if dti < 45: rules_passed += 1
    if dscr > 1.2: rules_passed += 1
    if app_data['credit_score'] > 650: rules_passed += 1
    if pd_score < 25: rules_passed += 1
    
    st.write("### Rule Checks")
    st.checkbox("DTI < 45%", value=(dti < 45), disabled=True)
    st.checkbox("DSCR > 1.2x", value=(dscr > 1.2), disabled=True)
    st.checkbox("Credit Score > 650", value=(app_data['credit_score'] > 650), disabled=True)
    st.checkbox("ML PD Score < 25%", value=(pd_score < 25), disabled=True)
    
    st.divider()
    if rules_passed >= 3:
        st.success(f"✅ APPROVED - Risk Grade: {risk_grade}")
        st.write(f"**Recommended Max Loan:** ₹ {app_data['annual_income'] * 1.5:,.2f}")
    else:
        st.error(f"❌ DECLINED - Risk Grade: {risk_grade}")
        st.write("Fails to meet minimum bank underwriting criteria combining ML risk assessment and financial ratios.")
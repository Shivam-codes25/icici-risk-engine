# database_builder.py
import sqlite3
import pandas as pd
import numpy as np

def generate_synthetic_data(num_records=1000):
    np.random.seed(42)
    
    # 1. Applicants Data
    applicant_ids = [f"APP{1000+i}" for i in range(num_records)]
    incomes = np.random.normal(1500000, 500000, num_records).clip(300000, 5000000)
    debts = incomes * np.random.uniform(0.1, 0.6, num_records)
    loans = np.random.normal(2000000, 800000, num_records).clip(500000, 10000000)
    scores = np.random.normal(720, 50, num_records).clip(300, 900)
    
    # Simulate realistic defaults (lower score & higher debt = higher default chance)
    z = 4 + (scores * -0.01) + ((debts/incomes) * 5)
    probs = 1 / (1 + np.exp(-z))
    defaults = np.random.binomial(1, probs)

    applicants = pd.DataFrame({
        'applicant_id': applicant_ids,
        'age': np.random.randint(22, 60, num_records),
        'annual_income': incomes,
        'existing_debt': debts,
        'loan_amount': loans,
        'interest_rate': np.random.uniform(8.5, 14.5, num_records),
        'tenure_months': np.random.choice([36, 48, 60, 72], num_records),
        'credit_score': scores,
        'default_history': np.random.choice([0, 1], num_records, p=[0.85, 0.15]),
        'default': defaults
    })

    # 2. Financial Statements (For business applicants)
    financials = pd.DataFrame({
        'applicant_id': applicant_ids,
        'operating_income': incomes * np.random.uniform(0.3, 0.8, num_records),
        'annual_debt_service': debts * np.random.uniform(0.1, 0.3, num_records),
        'current_assets': incomes * np.random.uniform(0.5, 1.5, num_records),
        'current_liabilities': debts * np.random.uniform(0.5, 1.2, num_records),
        'total_debt': debts,
        'shareholder_equity': incomes * np.random.uniform(0.5, 2.0, num_records),
        'collateral_value': loans * np.random.uniform(0.8, 2.5, num_records)
    })

    # 3. Forex Exposure
    forex = pd.DataFrame({
        'applicant_id': applicant_ids,
        'usd_receivables': np.random.exponential(500000, num_records),
        'usd_payables': np.random.exponential(300000, num_records)
    })

    return applicants, financials, forex

def build_database():
    print("Generating synthetic data...")
    app_df, fin_df, fx_df = generate_synthetic_data()
    
    print("Building SQLite Database...")
    conn = sqlite3.connect('icici_risk.db')
    
    app_df.to_sql('applicants', conn, if_exists='replace', index=False)
    fin_df.to_sql('financials', conn, if_exists='replace', index=False)
    fx_df.to_sql('forex', conn, if_exists='replace', index=False)
    
    conn.close()
    print("Database 'icici_risk.db' created successfully!")

if __name__ == "__main__":
    build_database()
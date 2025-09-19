# app.py
"""
Smart Expense Tracker & Financial Advisor (Production-ready 2025)
---------------------------------------------------------------
Features:
- Upload CSV or PDF bank statement
- Auto-categorize (rules + ML classifier) or use existing categories
- Monthly summaries, category breakdowns, budget (50/30/20 rule)
- Savings tips, predictions, and downloadable reports (CSV/Excel)
"""

import streamlit as st
import pandas as pd
import numpy as np
import io, re
from datetime import datetime
from io import StringIO
import matplotlib.pyplot as plt

# ML imports
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# PDF parsing
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# ----------------------------------------------------------------------
# App Configuration
# ----------------------------------------------------------------------
st.set_page_config(page_title="Smart Expense Tracker", layout="wide")
st.title("💸 Smart Expense Tracker & Financial Advisor")
st.markdown("Upload a **CSV or PDF** of your bank transactions → auto-categorize, analyze, and get savings tips.")

# ----------------------------------------------------------------------
# Built-in Sample Data
# ----------------------------------------------------------------------
SAMPLE_CSV = """Date,Description,Amount,Type,Category
2025-08-01,Salary August,50000,Credit,salary
2025-08-02,Starbucks Coffee,-4.5,Debit,dining
2025-08-03,Uber Ride,-12.0,Debit,transport
2025-08-05,Electricity Bill - BSES,-30.0,Debit,utilities
2025-08-07,Netflix Subscription,-8.99,Debit,entertainment
2025-08-09,Amazon Purchase - Books,-25.0,Debit,shopping
"""

# ----------------------------------------------------------------------
# Keyword Rules
# ----------------------------------------------------------------------
KEYWORD_MAP = {
    ('salary','payroll','deposit'): 'salary',
    ('rent',): 'rent',
    ('electricity','water','bill','recharge','mobile','phone'): 'utilities',
    ('uber','ola','taxi','metro','train','flight','hotel','travel','ticket'): 'transport',
    ('coffee','restaurant','dining','pizza','cafe','swiggy','zomato','food'): 'dining',
    ('amazon','flipkart','shopping','myntra'): 'shopping',
    ('netflix','spotify','prime','subscription','hotstar'): 'entertainment',
    ('grocery','supermarket','bazaar','mart'): 'groceries',
    ('pharmacy','clinic','hospital','doctor','medical','medicine'): 'healthcare',
    ('insurance','premium','policy'): 'insurance',
    ('interest','dividend'): 'interest',
    ('transfer','neft','imps','upi'): 'transfer',
    ('atm','withdrawal','cash'): 'withdrawal',
    ('fee','charges','fine','penalty'): 'fees',
    ('tax','tds','gst'): 'taxes',
    ('gym','fitness','workout','membership'): 'fitness',
    ('investment','mutual fund','stocks','shares','sip'): 'investment',
}

def keyword_category(desc: str):
    if not isinstance(desc, str):
        return None
    d = desc.lower()
    for kws, cat in KEYWORD_MAP.items():
        if any(kw in d for kw in kws):
            return cat
    return None

# ----------------------------------------------------------------------
# Load Data (CSV/PDF)
# ----------------------------------------------------------------------
def load_csv(uploaded_bytes):
    try:
        return pd.read_csv(io.BytesIO(uploaded_bytes))
    except Exception:
        try:
            return pd.read_csv(io.StringIO(uploaded_bytes.decode('utf-8', errors='ignore')))
        except Exception:
            st.error("❌ Could not parse CSV. Ensure it has headers.")
            return None

def parse_pdf(uploaded_bytes):
    if not PDFPLUMBER_AVAILABLE:
        st.warning("⚠️ PDF parsing requires `pdfplumber`. Install or upload a CSV instead.")
        return None
    rows = []
    with pdfplumber.open(io.BytesIO(uploaded_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                for line in txt.split('\n'):
                    d = re.search(r'(\d{4}[/-]\d{2}[/-]\d{2}|\d{2}[/-]\d{2}[/-]\d{4})', line)
                    a = re.search(r'(-?\d+[.,]?\d*)$', line)
                    if d and a:
                        desc = line.replace(d.group(0),'').replace(a.group(0),'').strip()
                        try:
                            amt = float(a.group(0).replace(',',''))
                        except:
                            continue
                        rows.append((d.group(0), desc, amt))
    if not rows:
        st.warning("⚠️ No transactions found in PDF. Try CSV instead.")
        return None
    return pd.DataFrame(rows, columns=['Date','Description','Amount'])

def standardize_df(df):
    mapping = {}
    for c in df.columns:
        lc = c.lower()
        if 'date' in lc: mapping[c] = 'Date'
        elif 'desc' in lc: mapping[c] = 'Description'
        elif 'amount' in lc or 'amt' in lc: mapping[c] = 'Amount'
        elif 'type' in lc: mapping[c] = 'Type'
        elif 'category' in lc: mapping[c] = 'Category'
    df = df.rename(columns=mapping)

    # Clean Amount
    if 'Amount' in df.columns:
        df['Amount'] = df['Amount'].astype(str).str.replace(',','').str.replace('₹','')
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    # Debit/Credit handling
    if 'Type' in df.columns:
        df['Amount'] = df.apply(lambda r: abs(r['Amount']) if 'credit' in str(r['Type']).lower() else -abs(r['Amount']), axis=1)

    # Dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    return df.dropna(subset=['Date','Description','Amount'])

# ----------------------------------------------------------------------
# ML Model Training
# ----------------------------------------------------------------------
@st.cache_resource
def get_base_model():
    sample_df = pd.read_csv(StringIO(SAMPLE_CSV))
    X = sample_df['Description'].fillna("unknown")
    y = sample_df['Category']
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1,2)), LogisticRegression(max_iter=1000))
    model.fit(X, y_enc)
    return model, le

model, label_encoder = get_base_model()

def predict_category(desc):
    rule = keyword_category(desc)
    if rule: return rule
    try:
        pred = model.predict([str(desc)])[0]
        return label_encoder.inverse_transform([pred])[0]
    except:
        return "others"

# ----------------------------------------------------------------------
# Main App Logic
# ----------------------------------------------------------------------
st.sidebar.header("Upload your statement")
uploaded = st.sidebar.file_uploader("CSV or PDF", type=['csv','pdf'])
use_sample = st.sidebar.checkbox("Use sample demo data", value=False)

if use_sample:
    df_raw = pd.read_csv(StringIO(SAMPLE_CSV))
elif uploaded:
    content = uploaded.read()
    if uploaded.name.endswith(".pdf"):
        df_raw = parse_pdf(content)
    else:
        df_raw = load_csv(content)
    if df_raw is None:
        st.stop()
else:
    st.info("⬅️ Upload a statement (CSV preferred) or check *sample demo data*.")
    st.stop()

df = standardize_df(df_raw)
if df.empty:
    st.error("❌ Could not extract valid data. Ensure columns like Date, Description, Amount exist.")
    st.stop()

st.write("### Preview")
st.dataframe(df.head())

# Categorization
if 'Category' not in df.columns:
    df['Category'] = df['Description'].apply(predict_category)

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
income = df[df['Amount'] > 0]['Amount'].sum()
expense = -df[df['Amount'] < 0]['Amount'].sum()
net = income - expense
col1, col2, col3 = st.columns(3)
col1.metric("Income", f"{income:,.2f}")
col2.metric("Expense", f"{expense:,.2f}")
col3.metric("Net", f"{net:,.2f}")

# ----------------------------------------------------------------------
# Expense Breakdown
# ----------------------------------------------------------------------
st.write("### Expense Breakdown by Category")
cat_sum = df[df['Amount']<0].groupby('Category')['Amount'].sum().abs().sort_values(ascending=False)
st.bar_chart(cat_sum)

# ----------------------------------------------------------------------
# Monthly Trend
# ----------------------------------------------------------------------
st.write("### Monthly Trend")
df['Month'] = df['Date'].dt.to_period("M")
monthly = df.groupby('Month')['Amount'].sum()
st.line_chart(monthly)

# ----------------------------------------------------------------------
# Budget Insights (50/30/20 Rule)
# ----------------------------------------------------------------------
st.write("### Budget Insights (50/30/20 Rule)")
needs = cat_sum.get('utilities',0) + cat_sum.get('groceries',0) + cat_sum.get('rent',0) + cat_sum.get('healthcare',0)
wants = cat_sum.get('entertainment',0) + cat_sum.get('shopping',0) + cat_sum.get('dining',0) + cat_sum.get('transport',0)
savings = max(income - (needs+wants),0)

st.progress(min(needs/income if income>0 else 0,1.0))
st.caption(f"Needs: {needs:.2f} ({(needs/income*100 if income>0 else 0):.1f}%)")

st.progress(min(wants/income if income>0 else 0,1.0))
st.caption(f"Wants: {wants:.2f} ({(wants/income*100 if income>0 else 0):.1f}%)")

st.progress(min(savings/income if income>0 else 0,1.0))
st.caption(f"Savings: {savings:.2f} ({(savings/income*100 if income>0 else 0):.1f}%)")

# ----------------------------------------------------------------------
# Savings Tips
# ----------------------------------------------------------------------
st.write("### Savings Tips")
if expense > income:
    st.warning("⚠️ Spending exceeds income. Cut back immediately.")
else:
    st.success("✅ Spending is under control.")
if "entertainment" in cat_sum.index and cat_sum["entertainment"] > 100:
    st.write("- Reduce subscriptions or dining out to save money.")
if "transport" in cat_sum.index and cat_sum["transport"] > 200:
    st.write("- Consider cheaper travel options.")
if savings < 0.2*income:
    st.write("- Try to save at least 20% of your income each month.")

# ----------------------------------------------------------------------
# Downloads
# ----------------------------------------------------------------------
st.download_button("⬇️ Download categorized CSV", df.to_csv(index=False).encode('utf-8'),
                   "categorized_expenses.csv", "text/csv")
st.download_button("⬇️ Download Excel", df.to_excel("expenses.xlsx", index=False), "expenses.xlsx")

st.write("### All Transactions")
st.dataframe(df)

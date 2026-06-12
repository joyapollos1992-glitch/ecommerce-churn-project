import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. SET PAGE CONFIG (Must be first) ---
st.set_page_config(page_title="ChurnPredict AI", layout="wide")

# --- 2. LOAD ASSETS (Cached for 4GB RAM Efficiency) ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scalar.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found! Please run train_model.py first.")
    st.stop()

# --- 3. UI HEADER ---
st.title("📊 E-Commerce Customer Churn Predictor")
st.markdown("""
Predict which customers are likely to stop shopping with us using Machine Learning.
*System Status: Optimized for Low-Memory Environments.*
""")

# --- 4. SIDEBAR INPUTS (System Design: Single Prediction) ---
st.sidebar.header("Customer Profile Input")

def get_user_input():
    tenure = st.sidebar.slider("Tenure (Months)", 1, 60, 12)
    cs_calls = st.sidebar.number_input("Customer Service Calls", 0, 20, 2)
    complains = st.sidebar.selectbox("Past Complains?", ["No", "Yes"])
    discount = st.sidebar.slider("Discount Usage (0-1.0)", 0.0, 1.0, 0.2)
    frequency = st.sidebar.number_input("Purchase Frequency", 1, 100, 5)
    recency = st.sidebar.number_input("Recency (Days since last buy)", 1, 365, 30)
    
    # Map 'Yes/No' to 1/0
    complains_val = 1 if complains == "Yes" else 0
    
    # Create DataFrame for prediction
    data = {
        'tenure_months': tenure,
        'cs_calls': cs_calls,
        'complains': complains_val,
        'discount_usage': discount,
        'frequency': frequency,
        'recency_days': recency
    }
    return pd.DataFrame([data])

input_df = get_user_input()

# --- 5. MAIN DASHBOARD AREA ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Customer Data Summary")
    st.write(input_df)

    # --- PREDICTION LOGIC ---
    if st.button("Analyze Churn Risk"):
        # A. Scale the input using the saved scalar
        input_scaled = scaler.transform(input_df)
        
        # B. Make Prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] # Probability of Churn (Class 1)
        
        # C. Display Results
        st.divider()
        if prediction == 1:
            st.error(f"### 🚩 High Risk of Churn! ({probability:.1%} probability)")
            st.warning("Recommendation: Send a retention discount or loyalty offer immediately.")
        else:
            st.success(f"### ✅ Low Risk Customer ({probability:.1%} probability)")
            st.info("Recommendation: Maintain current engagement.")

with col2:
    st.subheader("Feature Impact")
    # Simple static explanation for the project demo
    st.info("""
    **Key Indicators:**
    1. **Recency:** High days = High risk.
    2. **CS Calls:** More calls = Frustration.
    3. **Tenure:** Long-term users stay.
    """)
    
    # Add a visual "Risk Meter" using a progress bar
    if 'probability' in locals():
        st.write("Churn Probability Meter")
        st.progress(float(probability))

# --- 6. FOOTER ---
st.divider()
st.caption("Final Year Project: Predictive Analytics for E-commerce Churn | RAM Status: Stable")
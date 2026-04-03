import streamlit as st
import pandas as pd
from src.model import load_data, train_model

# Load and train
data = load_data("data/loan_data.csv")
model, columns = train_model(data)

st.title("🏠 Loan Eligibility Prediction")

income = st.number_input("Applicant Income")
loan = st.number_input("Loan Amount")
credit = st.selectbox("Credit History", [0, 1])

if st.button("Predict"):
    sample = pd.DataFrame([[income, loan, credit]], columns=columns[:3])
    result = model.predict(sample)

    if result[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")
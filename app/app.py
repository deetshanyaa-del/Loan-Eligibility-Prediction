import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.model import load_data, train_model

# Load data
data = load_data("data/loan_data.csv")
model, columns = train_model(data)

st.set_page_config(page_title="Loan Prediction", layout="centered")

st.title("🏠 Loan Eligibility Prediction")
st.markdown("### Enter Applicant Details")

# -----------------------------
# INPUT FIELDS (FULL VERSION)
# -----------------------------

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

income = st.number_input("Applicant Income", min_value=0)
co_income = st.number_input("Coapplicant Income", min_value=0)
loan = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term", min_value=0)

credit = st.selectbox("Credit History", [0, 1])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Loan Status"):
    try:
        input_data = pd.DataFrame([0]*len(columns)).T
        input_data.columns = columns

        # Fill values
        input_data["ApplicantIncome"] = income
        input_data["CoapplicantIncome"] = co_income
        input_data["LoanAmount"] = loan
        input_data["Loan_Amount_Term"] = loan_term
        input_data["Credit_History"] = credit

        # Categorical encoding manually
        input_data["Gender_Male"] = 1 if gender == "Male" else 0
        input_data["Married_Yes"] = 1 if married == "Yes" else 0
        input_data["Education_Not Graduate"] = 1 if education == "Not Graduate" else 0
        input_data["Self_Employed_Yes"] = 1 if self_employed == "Yes" else 0

        if dependents == "1":
            input_data["Dependents_1"] = 1
        elif dependents == "2":
            input_data["Dependents_2"] = 1
        elif dependents == "3+":
            input_data["Dependents_3+"] = 1

        if property_area == "Urban":
            input_data["Property_Area_Urban"] = 1
        elif property_area == "Semiurban":
            input_data["Property_Area_Semiurban"] = 1

        # Predict
        result = model.predict(input_data)

        if result[0] == 1:
            st.success("✅ Loan Approved 🎉")
        else:
            st.error("❌ Loan Rejected")

    except Exception as e:
        st.error(f"Error: {e}")
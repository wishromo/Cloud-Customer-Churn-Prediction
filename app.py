import streamlit as st
import pandas as pd
import joblib

model = joblib.load("cloud_churn_final_pipeline.pkl")
threshold = joblib.load("cloud_churn_threshold.pkl")

st.title("Cloud Customer Churn Prediction")

st.write(
    "Enter customer details below to estimate the probability of churn."
)

monthly_spend = st.number_input(
    "Monthly Cloud Spend",
    min_value=0.0,
    value=300.0
)

compute_hours = st.number_input(
    "Compute Usage Hours",
    min_value=0.0,
    value=150.0
)

storage_gb = st.number_input(
    "Storage GB",
    min_value=0.0,
    value=400.0
)

support_tickets = st.number_input(
    "Support Tickets",
    min_value=0,
    value=2
)

account_age = st.number_input(
    "Account Age (Months)",
    min_value=1,
    value=12
)

contract_type = st.selectbox(
    "Contract Type",
    ["Monthly", "Annual", "Two-Year"]
)

region = st.selectbox(
    "Region",
    ["North America", "Europe", "Asia Pacific", "Middle East"]
)

services_used = st.number_input(
    "Services Used",
    min_value=1,
    value=3
)

downtime_hours = st.number_input(
    "Downtime Hours",
    min_value=0.0,
    value=1.0
)

auto_renew = st.selectbox(
    "Auto Renew",
    ["Yes", "No"]
)

customer_satisfaction = st.slider(
    "Customer Satisfaction",
    1,
    5,
    3
)

if st.button("Predict Churn"):

    customer = pd.DataFrame([{
        "Monthly_Cloud_Spend": monthly_spend,
        "Compute_Usage_Hours": compute_hours,
        "Storage_GB": storage_gb,
        "Support_Tickets": support_tickets,
        "Account_Age_Months": account_age,
        "Contract_Type": contract_type,
        "Region": region,
        "Services_Used": services_used,
        "Downtime_Hours": downtime_hours,
        "Auto_Renew": auto_renew,
        "Customer_Satisfaction": customer_satisfaction
    }])

    probability = model.predict_proba(customer)[:, 1][0]

    prediction = (
        "Yes"
        if probability >= threshold
        else "No"
    )

    st.subheader("Prediction")

    st.write(
        f"Churn Probability: {probability:.2%}"
    )

    st.write(
        f"Predicted Churn: **{prediction}**"
    )
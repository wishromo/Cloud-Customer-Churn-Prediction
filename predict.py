import joblib
import pandas as pd
import numpy as np

model = joblib.load("models/cloud_churn_final_pipeline.pkl")
threshold = joblib.load("models/cloud_churn_threshold.pkl")

new_customer = pd.DataFrame([{
    "Monthly_Cloud_Spend": 450,
    "Compute_Usage_Hours": 210,
    "Storage_GB": 600,
    "Support_Tickets": 4,
    "Account_Age_Months": 10,
    "Contract_Type": "Monthly",
    "Region": "Europe",
    "Services_Used": 3,
    "Downtime_Hours": 4.2,
    "Auto_Renew": "No",
    "Customer_Satisfaction": 2
}])

probability = model.predict_proba(new_customer)[:, 1][0]
prediction = "Yes" if probability >= threshold else "No"

print("Churn probability:", round(probability, 3))
print("Prediction:", prediction)
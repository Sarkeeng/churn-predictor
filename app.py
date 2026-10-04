import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_model.pkl")
st.title("Customer Churn Predictor")

yn = ["Yes", "No"]
c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", yn)
    dependents = st.selectbox("Dependents", yn)
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", yn)
    multi = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
with c2:
    opt = ["No", "Yes", "No internet service"]
    security = st.selectbox("Online Security", opt)
    backup = st.selectbox("Online Backup", opt)
    device = st.selectbox("Device Protection", opt)
    tech = st.selectbox("Tech Support", opt)
    tv = st.selectbox("Streaming TV", opt)
    movies = st.selectbox("Streaming Movies", opt)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", yn)
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly Charges ($)", 15.0, 130.0, 70.0)

if st.button("Predict churn risk"):
    row = pd.DataFrame([{
        "Gender": gender, "Senior Citizen": senior, "Partner": partner,
        "Dependents": dependents, "Tenure Months": tenure, "Phone Service": phone,
        "Multiple Lines": multi, "Internet Service": internet,
        "Online Security": security, "Online Backup": backup,
        "Device Protection": device, "Tech Support": tech, "Streaming TV": tv,
        "Streaming Movies": movies, "Contract": contract,
        "Paperless Billing": paperless, "Payment Method": payment,
        "Monthly Charges": monthly, "Total Charges": monthly * tenure}])
    p = model.predict_proba(row)[0, 1]
    st.metric("Churn probability", f"{p:.0%}")
    if p >= 0.5:
        st.error("High risk: this customer is likely to leave.")
    else:
        st.success("Lower risk: this customer is likely to stay.")
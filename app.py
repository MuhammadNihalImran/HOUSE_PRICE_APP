import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load data and model
data = pd.read_csv("./data/clean_Bengaluru_House_Data.csv")
locations = sorted(data["location"].unique())
pipe = joblib.load("model.pkl")

# App title
st.title("🏠 House Price Predictor")

# Input form
with st.form("prediction_form"):
    location = st.selectbox("Select Location", locations)
    total_sqft = st.number_input("Total Square Feet", min_value=100.0, step=10.0)
    bath = st.number_input("Number of Bathrooms", min_value=1, step=1)
    bhk = st.number_input("BHK (Bedrooms)", min_value=1, step=1)

    submit = st.form_submit_button("Predict Price")

# Prediction
if submit:
    inputs = pd.DataFrame(
        [[location, total_sqft, bath, bhk]],
        columns=["location", "total_sqft", "bath", "bhk"],
    )

    price = pipe.predict(inputs)[0] * 1e5  # Convert from lakhs
    st.success(f"💰 Estimated Price: ₹{round(price, 2)}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model & scaler
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🌤️ Forecasting Temperature dengan Random Forest")
st.write("Aplikasi prediksi temperatur berbasis time series.")

# Input untuk fitur
humidity = st.number_input("Humidity", 0.0, 1.0, 0.5)
pressure = st.number_input("Pressure (millibars)", 900.0, 1100.0, 1010.0)
visibility = st.number_input("Visibility (km)", 0.0, 20.0, 10.0)
wind_speed = st.number_input("Wind Speed (km/h)", 0.0, 50.0, 5.0)
apparent_temp = st.number_input("Apparent Temperature (C)", -20.0, 50.0, 20.0)

# LAG fitur minimal (dummy)
lag_values = {}
for lag in range(1, 25):
    lag_values[f"Temperature (C)_lag{lag}"] = st.number_input(
        f"Suhu (lag {lag})", -20.0, 50.0, 20.0
    )

if st.button("Prediksi Suhu"):
    # Buat dataframe input sesuai urutan training
    input_values = [
        humidity,
        pressure,
        visibility,
        wind_speed,
        apparent_temp
    ] + list(lag_values.values())

    input_df = pd.DataFrame([input_values])

    # Scaling
    scaled_input = scaler.transform(input_df)

    # Predict
    pred = rf_model.predict(scaled_input)[0]

    st.success(f"Prediksi Suhu: **{pred:.2f} °C**")

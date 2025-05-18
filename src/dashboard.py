import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load models
nn_model = load_model('models/neural_network_model.h5')  # Replace with actual path to your Keras model
rf_model = joblib.load('models/random_forest_model.pkl')  # Replace with actual path to your Random Forest model

# Streamlit App
st.title('Air Quality Prediction Dashboard')
st.markdown('This dashboard predicts air quality based on environmental factors.')

# Input data form
st.sidebar.header("Enter Environmental Data")

# Inputs for prediction
temp = st.sidebar.number_input("Temperature (°C)", min_value=-30, max_value=50, value=25)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0, max_value=100, value=60)
co2_level = st.sidebar.number_input("CO2 Level (ppm)", min_value=0, max_value=10000, value=400)
no2_level = st.sidebar.number_input("NO2 Level (ppm)", min_value=0, max_value=1000, value=30)
pm25_level = st.sidebar.number_input("PM2.5 Level (µg/m³)", min_value=0, max_value=500, value=12)

# Preprocess the input data to match the model's expected format
input_data = np.array([[temp, humidity, co2_level, no2_level, pm25_level]])

# Prediction button
if st.sidebar.button("Predict"):
    # Get predictions from both models
    nn_preds = nn_model.predict(input_data)
    rf_preds = rf_model.predict(input_data)

    # Display the predictions
    st.subheader("Predictions")
    st.write(f"Neural Network Prediction: {nn_preds[0][0]:.2f} AQI")
    st.write(f"Random Forest Prediction: {rf_preds[0]:.2f} AQI")

    # Visualize the results (optional)
    st.subheader("Prediction Comparison")
    st.bar_chart([nn_preds[0][0], rf_preds[0]], width=0, height=0, use_container_width=True)

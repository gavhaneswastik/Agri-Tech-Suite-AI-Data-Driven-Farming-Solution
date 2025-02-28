import pickle
import streamlit as st
import numpy as np

# Load the trained model
with open("crop_recommendation_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the label encoder (if used)
with open("label_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)
# Streamlit app
st.title("Crop Recommendation System")

# Input fields for the features
N = st.number_input("Nitrogen content (N)", min_value=0, max_value=140, step=1)
P = st.number_input("Phosphorus content (P)", min_value=0, max_value=140, step=1)
K = st.number_input("Potassium content (K)", min_value=0, max_value=200, step=1)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)

# Predict button
if st.button("Predict"):
    # Prepare the feature array
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Make the prediction
    prediction = model.predict(features)
    
    # Decode the prediction (if label encoder is used)
    try:
        decoded_prediction = encoder.inverse_transform(prediction)
        st.success(f"Recommended Crop: {decoded_prediction[0]}")
    except:
        st.error("Error in decoding the prediction. Check your label encoder or data.")

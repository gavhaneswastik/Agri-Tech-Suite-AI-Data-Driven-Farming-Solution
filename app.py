import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import requests
from PIL import Image

# OpenWeatherMap API Key
API_KEY = "30d4741c779ba94c470ca1f63045390a"

# Load models and required files
@st.cache_resource
def load_models():
    plant_disease_model = tf.keras.models.load_model("D:/my data/Final Year PRoject/Plant_Disease/plant_disease_prediction_model.keras")
    class_indices = json.load(open("D:/my data/Final Year PRoject/Plant_Disease/class_indices.json"))
    crop_model = joblib.load("D:/my data/Final Year PRoject/Plant_Recommendation/crop_recommendation_model.pkl")
    crop_encoder = joblib.load("D:/my data/Final Year PRoject/Plant_Recommendation/label_encoder.pkl")
    fertilizer_model = joblib.load("D:/my data/Final Year PRoject/Fertilizer_Recommendation/fertilizer_prediction_model.pkl")
    return plant_disease_model, class_indices, crop_model, crop_encoder, fertilizer_model

# Load models
plant_disease_model, class_indices, crop_model, crop_encoder, fertilizer_model = load_models()

# Function to get weather data
def get_weather(city):
    """Fetches weather data for a given city."""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if data.get("cod") != 200:
        return None, None  # City not found

    weather = data["weather"][0]["description"].title()
    temp = round(data["main"]["temp"], 1)
    return weather, temp

# Modules
def plant_disease_prediction():
    st.title("🌱 Plant Disease Prediction")
    uploaded_image = st.file_uploader("Upload an image of the plant", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Classify Disease"):
            img_array = np.array(image.resize((224, 224))).astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = plant_disease_model.predict(img_array)
            predicted_class = class_indices[str(np.argmax(prediction, axis=1)[0])]
            st.success(f"Prediction: {predicted_class}")

def crop_recommendation():
    st.title("🌾 Crop Recommendation")
    col1, col2 = st.columns(2)
    N = col1.number_input("Nitrogen (N)", min_value=0, max_value=140, step=1)
    P = col1.number_input("Phosphorus (P)", min_value=0, max_value=140, step=1)
    K = col1.number_input("Potassium (K)", min_value=0, max_value=200, step=1)
    temperature = col2.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1)
    humidity = col2.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
    ph = col2.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1)
    rainfall = col2.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)
    
    if st.button("Recommend Crop"):
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = crop_model.predict(features)
        recommended_crop = crop_encoder.inverse_transform(prediction)[0]
        st.success(f"Recommended Crop: {recommended_crop}")

def fertilizer_recommendation():
    st.title("🧪 Fertilizer Recommendation")
    col1, col2 = st.columns(2)
    temperature = col1.number_input("Temperature (°C)", min_value=0, max_value=50)
    humidity = col1.number_input("Humidity (%)", min_value=0, max_value=100)
    moisture = col1.number_input("Moisture (%)", min_value=0, max_value=100)
    soil_type = col2.selectbox("Soil Type", ["Sandy", "Loamy", "Black Cotton", "Clayey"])
    crop_type = col2.selectbox("Crop Type", ["Maize", "Wheat", "Rice", "Sugarcane"])
    nitrogen = col2.number_input("Nitrogen", min_value=0, max_value=100)
    potassium = col2.number_input("Potassium", min_value=0, max_value=100)
    phosphorous = col2.number_input("Phosphorous", min_value=0, max_value=100)
    
    if st.button("Predict Fertilizer"):
        input_data = pd.DataFrame({
            'Temperature': [temperature],
            'Humidity': [humidity],
            'Moisture': [moisture],
            'Soil Type': [soil_type],
            'Crop Type': [crop_type],
            'Nitrogen': [nitrogen],
            'Potassium': [potassium],
            'Phosphorous': [phosphorous]
        })
        prediction = fertilizer_model.predict(input_data)[0]
        st.success(f"Recommended Fertilizer: {prediction}")

def weather_forecast():
    st.title("🌤 Weather Forecasting")
    city = st.text_input("Enter city name:", "New York")
    
    if st.button("Get Weather"):
        weather, temp = get_weather(city)
        
        if weather is None:
            st.error("❌ City not found. Please enter a valid city.")
        else:
            st.success(f"🌍 City: {city}\n🌡 Temperature: {temp}°C\n☁ Condition: {weather}")

# Streamlit App
#st.set_page_config(page_title="Agritech Suite", page_icon="🌿", layout="wide")
st.sidebar.title("🌿 Agritech Suite")
st.sidebar.write("Select a module to get started:")

# Sidebar options
options = {
    "🌱 Plant Disease Prediction": plant_disease_prediction,
    "🌾 Crop Recommendation": crop_recommendation,
    "🧪 Fertilizer Recommendation": fertilizer_recommendation,
    "🌤 Weather Forecasting": weather_forecast,
}

choice = st.sidebar.radio("Choose a module:", list(options.keys()))
options[choice]()  # Call the selected module
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from utils import predict_disease, recommend_fertilizer

model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

st.title(" Plant Disease Detection & Fertilizer Recommendation App ")

st.header(" Upload a Plant Leaf Image")
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')  # Ensure RGB format
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")

        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.uint8)
        if img_array.size == 0:
            st.error("Error: Could not read the uploaded image. Please try again.")
        else:
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            disease = predict_disease(img_array, model) #Pass Model to predict_disease
            if disease:
                st.success(f" Disease Detected: {disease}")
            else:
                st.error("Could not classify the image. Please try again or upload a different image.")

            if disease:
                st.header(" Enter Environmental and Soil Details for Fertilizer Recommendation")

                temperature = st.number_input("️ Temperature (°C)", min_value=0, max_value=50)
                humidity = st.number_input(" Humidity (%)", min_value=0, max_value=100)
                moisture = st.number_input("🪴 Soil Moisture (%)", min_value=0, max_value=100)
                soil_type = st.selectbox(" Soil Type", ['Sandy', 'Loamy', 'Clay'])
                crop_type = st.selectbox(" Crop Type", ['Maize', 'Wheat', 'Rice'])
                nitrogen = st.number_input(" Nitrogen Level", min_value=0)
                potassium = st.number_input("⚗️ Potassium Level", min_value=0)
                phosphorous = st.number_input(" Phosphorous Level", min_value=0)

                if st.button(" Recommend Fertilizer"):
                    user_data = {
                        'Temperature': temperature,
                        'Humidity': humidity,
                        'Moisture': moisture,
                        'Soil Type': soil_type,
                        'Crop Type': crop_type,
                        'Nitrogen': nitrogen,
                        'Potassium': potassium,
                        'Phosphorous': phosphorous
                    }
                    fertilizer = recommend_fertilizer(user_data)
                    st.success(f"✅ Recommended Fertilizer: {fertilizer}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
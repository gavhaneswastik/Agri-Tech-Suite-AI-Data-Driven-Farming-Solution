import cv2
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

# Load Models
disease_model = tf.keras.models.load_model('E:\Final Year PRoject\Plant Disease + Fertilizer\models\disease_model.h5')
fertilizer_model = joblib.load('E:/Final Year PRoject/Plant Disease + Fertilizer/models/recommend_fertilizer_model.pkl')

# Preprocess Image for Disease Detection
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Predict Plant Disease
def predict_disease(image):
    processed_img = preprocess_image(image)
    predictions = disease_model.predict(processed_img)
    class_idx = np.argmax(predictions)
    disease_classes = ['Healthy', 'Bacterial Spot', 'Leaf Mold']  # Example
    return disease_classes[class_idx]

# Predict Fertilizer
def recommend_fertilizer(user_input):
    df = pd.DataFrame([user_input])
    prediction = fertilizer_model.predict(df)
    fertilizer_classes = ['Urea', 'DAP', 'Potash']  # Example
    return fertilizer_classes[prediction[0]]

import streamlit as st
import pandas as pd
import joblib

def main():
    st.title('Fertilizer Recommendation System')
    
    # Create input fields for all features
    temperature = st.number_input('Temperature', min_value=0, max_value=50)
    humidity = st.number_input('Humidity', min_value=0, max_value=100)
    moisture = st.number_input('Moisture', min_value=0, max_value=100)
    
    # Categorical inputs
    soil_type = st.selectbox('Soil Type', ['Sandy', 'Loamy', 'Black Cotton', 'Clayey'])
    crop_type = st.selectbox('Crop Type', ['Maize', 'Wheat', 'Rice', 'Sugarcane'])
    
    # Nutrient inputs
    nitrogen = st.number_input('Nitrogen', min_value=0, max_value=100)
    potassium = st.number_input('Potassium', min_value=0, max_value=100)
    phosphorous = st.number_input('Phosphorous', min_value=0, max_value=100)
    
    # Prediction button
    if st.button('Predict Fertilizer'):
        input_data = pd.DataFrame({
            'Temparature': [temperature],
            'Humidity ': [humidity],
            'Moisture': [moisture],
            'Soil Type': [soil_type],
            'Crop Type': [crop_type],
            'Nitrogen': [nitrogen],
            'Potassium': [potassium],
            'Phosphorous': [phosphorous]
        })
        
        # Load the model
        model = joblib.load('fertilizer_prediction_model.pkl')
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        st.success(f'Recommended Fertilizer: {prediction}')

if __name__ == '__main__':
    main()

import streamlit as st
import numpy as np
import pickle

# Load the model
model = pickle.load(open('NBClassifier.pkl', 'rb')) 

# Define crop array
crop_array = [
    'rice', 'maize', 'jute', 'cotton', 'coconut', 'papaya', 'orange', 'apple',
    'muskmelon', 'watermelon', 'grapes', 'mango', 'banana', 'pomegranate', 
    'lentil', 'blackgram', 'mungbean', 'mothbeans', 'pigeonpeas', 
    'kidneybeans', 'chickpea', 'coffee'
]

def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall):
    feature_list = [nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction = model.predict(single_pred)
    
    predicted_crop_name = prediction[0]

    if predicted_crop_name in crop_array:
        return f"{predicted_crop_name} is the best crop to be cultivated"
    else:
        return "No crop is predicted"

# Streamlit app
def main():
    st.title("Crop Prediction")

    # Input fields
    nitrogen = st.number_input("Nitrogen")
    phosphorus = st.number_input("Phosphorus")
    potassium = st.number_input("Potassium")
    temperature = st.number_input("Temperature")
    humidity = st.number_input("Humidity")
    pH = st.number_input("pH")
    rainfall = st.number_input("Rainfall")

    # Prediction button
    if st.button("Predict"):
        result = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall)
        st.write(result)

if __name__ == "__main__":
    main()

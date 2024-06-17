import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from PIL import Image 

st.title('Air Quality Predictor')
image = Image.open('Monitoring-station-ajax.jpg')
resized_image = image.resize((650, 450))
st.image(resized_image)

# Custom CSS for number input boxes
custom_css = """
<style>
.header-container {
    margin-bottom: 20px; /* Add margin at the bottom */
}

div.stNumberInput {
    margin-top: 20px; /* Add margin at the top */
}

div.stNumberInput input {
    background-color: #3A5FCD; /* Change to a different shade of blue */
    color: #000000; /* Text color (black) */
}
</style>
"""

st.markdown("""<div class="header-container">
                This app uses 4 inputs to predict air quality in cities around the world using a neural network model.
                Thanks to Zindi data! Use the form below to get started!
              </div>""", unsafe_allow_html=True)

# Add navigation links in the sidebar
st.sidebar.selectbox("Navigation", ["Home", "About", "Contact"])

model = pickle.load(open('random_forest_model_four_features.pkl', 'rb'))

cols=['precipitable_water_entire_atmosphere', 'relative_humidity_2m_above_ground', 
      'specific_humidity_2m_above_ground', 'temperature_2m_above_ground']    

# Define the prediction function
def predict(precipitable_water_entire_atmosphere, relative_humidity_2m_above_ground, 
            specific_humidity_2m_above_ground, temperature_2m_above_ground):
    
    prediction = model.predict([[precipitable_water_entire_atmosphere, relative_humidity_2m_above_ground, 
            specific_humidity_2m_above_ground, temperature_2m_above_ground]])
    
    return prediction


precipitable_water_entire_atmosphere = st.number_input('Precipitable water', 0.00, 100.00)
relative_humidity_2m_above_ground = st.number_input('Relative humidity', 0.00, 100.00)
specific_humidity_2m_above_ground = st.number_input('Specific humidity', 0.00, 100.00)
temperature_2m_above_ground = st.number_input('Temperature', 0.00, 100.00)

if st.button('Predict Air Quality'):
    a_q = predict(precipitable_water_entire_atmosphere, relative_humidity_2m_above_ground, 
            specific_humidity_2m_above_ground, temperature_2m_above_ground)
    st.success(f'The predicted air quality is {a_q[0]:.2f} PM2.5')











    
    
    
    
    
    
    
    
    
    
    
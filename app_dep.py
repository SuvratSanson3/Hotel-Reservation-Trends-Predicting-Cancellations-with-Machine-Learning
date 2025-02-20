import streamlit as st
import pickle 
import numpy as np
import pandas as pd
from PIL import Image

# Load the trained model
try:
    with open("final_model_xgb.pkl", 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Prediction function
def prediction(input_data):
    try:
        pred = model.predict_proba(input_data)[:,1][0]
        if pred > 0.5:
            return f"This booking is more likely to get canceled: Chances = {round(pred*100,2)}%"
        else:
            return f"This booking is less likely to get canceled: Chances = {round(pred*100,2)}%"
    except Exception as e:
        return f"Error during prediction: {e}"

# Main app
def main():
    st.title("INN Hotels - Cancellation Prediction")

    # Display hotel image
    image = Image.open("Hotel_image.jpeg")
    st.image(image, use_container_width=True)

    # User inputs
    lead_time = st.number_input("Enter Lead Time")
    market_dict = {'Online': 1, 'Offline': 0}
    market_segment_type = market_dict[st.selectbox('Enter the type of booking', ['Online', 'Offline'])]
    no_of_special_requests = st.selectbox("How many special requests have been made", [0, 1, 2, 3, 4, 5])
    avg_price_per_room = st.number_input("Enter the price per room")
    no_of_adults = st.selectbox('How many Adults', [0, 1, 2, 3, 4, 5, 6])
    no_of_weekend_nights = st.selectbox('How many weekend nights', [0, 1, 2, 3, 4, 5])
    required_car_parking_space = {'Yes': 1, 'No': 0}[st.selectbox('Does booking include parking facility?', ['Yes', 'No'])]
    no_of_week_nights = st.selectbox('How many week nights', [0, 1, 2, 3, 4, 5, 6, 7, 8])
    arrival_day = st.slider("What will be the day of arrival", 1, 31, 1)
    arrival_month = st.slider("What will be the month of arrival", 1, 12, 1)
    weekday_dict = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    arrival_weekday = weekday_dict[st.selectbox("Day of Arrival", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])]

    # Preparing input data for prediction
    input_data = [[lead_time, market_segment_type, no_of_special_requests,
                   avg_price_per_room, no_of_adults, no_of_weekend_nights,
                   required_car_parking_space, no_of_week_nights, arrival_day,
                   arrival_month, arrival_weekday]]

    # Predict and display the result
    if st.button("Predict"):
        response = prediction(input_data)
        st.success(response)

if __name__ == '__main__':
    main()

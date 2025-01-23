import streamlit as st
import numpy as np
import pickle

# Load the model
with open('final_model_xgb.pkl', 'rb') as file:
    model = pickle.load(file)

# Prediction function
def prediction(input_data):
    input_data = np.array(input_data, dtype='object')
    pred = model.predict_proba(input_data)[:, 1][0]
    if pred > 0.5:
        return f"This booking is more likely to get canceled: Chances = {round(pred * 100, 2)}%"
    else:
        return f"This booking is less likely to get canceled: Chances = {round(pred * 100, 2)}%"

# Main app
def main():
    st.title("INN Hotels - Cancellation Prediction")

    # User Inputs
    lt = st.number_input('Enter lead time (in days):', min_value=0, step=1)
    mkt = 1 if st.selectbox('Enter the type of booking:', ['Online', 'Offline']) == 'Online' else 0
    spcl = st.selectbox('How many special requests have been made?', [0, 1, 2, 3, 4, 5])
    price = st.number_input('Enter the price of the room:')
    adults = st.selectbox('How many adults per room?', [1, 2, 3, 4])
    wknd = st.number_input('How many weekend nights?', min_value=0, step=1)
    prk = 1 if st.selectbox('Does booking include parking facility?', ['Yes', 'No']) == 'Yes' else 0
    wk = st.number_input('How many weekday nights?', min_value=0, step=1)
    arr_d = st.slider('What will be the day of arrival?', min_value=1, max_value=31, step=1)
    arr_m = st.slider('What will be the month of arrival?', min_value=1, max_value=12, step=1)
    weekday_mapping = {'Mon': 0, 'Tues': 1, 'Wed': 2, 'Thurs': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    arr_wd = weekday_mapping[st.selectbox('What is the weekday of arrival?', list(weekday_mapping.keys()))]

    # Preparing input data
    input_list = [[lt, mkt, spcl, price, adults, wknd, prk, wk, arr_d, arr_m, arr_wd]]

    # Predict and Display
    if st.button('Predict'):
        response = prediction(input_list)
        st.success(response)

if __name__ == '__main__':
    main()

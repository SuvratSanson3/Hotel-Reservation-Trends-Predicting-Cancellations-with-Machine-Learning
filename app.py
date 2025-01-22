import streamlit as st
import pickle 
import numpy as np
import pandas as pd

with open("final_model_xgb.pkl",'rb') as file:
    model = pickle.load(file)

def prediction(input_data):
    pred=model.predict_proba(input_data)[:,1][0]

    if pred>0.5:
        return f"This booking is more likely to get canceled: Chances={round(pred,2)}"
    else:
        return f"This is booking is less likely to get canceled: Chances={round(pred,2)}"
    
def main():
    
    st.title("INN Hotels")

    lead_time=st.number_input("Enter Lead Time")
    market_dict={'Online':1,'Offline':0}
    market_segment_type= market_dict[st.selectbox('Enter the type of booking',['Online',"Offline"])]    
    no_of_special_requests=st.selectbox("How many special request have been made",[0,1,2,3,4,5])
    avg_price_per_room=st.number_input("Enter the price per room")
    no_of_adults=st.selectbox('How many Adults',[0,1,2,3,4,5,6])
    no_of_weekend_nights=st.selectbox('How many weekend nights',[0,1,2,3,4,5])
    required_car_parking_space=(lambda x:1 if x=='Yes' else 0)(st.selectbox('Does booking includes parking facility'),['Yes','No'])
    no_of_week_nights=st.selectbox('How many week Nights',[0,1,2,3,4,5,6,7,8])
    arrival_day=st.slider("What will be the day of arrival",1,31,1)
    arrival_month=st.slider("What will be the month of arrival",1,12,1)
    weekday_dict={'Mon':0,'Tue':1,'Wed':2,'Thur':3,'Fri':4,'Sat':5,'Sun':6}
    arrival_weekday=weekday_dict[st.selectbox("Day of Arrival",['Mon','Tue','Wed','Thur','Fri','Sat','Sun'])]


    input_data=[[lead_time, market_segment_type, no_of_special_requests,
               avg_price_per_room,no_of_adults,no_of_weekend_nights,
               required_car_parking_space,no_of_week_nights,arrival_day,
                arrival_month,arrival_weekday]]


    if st.button("predict"):
        response=prediction(input_data)
        st.success(response)


if __name__ == '__main__':
    main()

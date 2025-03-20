import streamlit as st
import requests

st.title('Loan Approval Prediction')

# Collect input data from user
age = st.number_input('Age', min_value=18, max_value=100)
annual_income = st.number_input('Annual Income', min_value=0)
monthly_expenses = st.number_input('Monthly Expenses', min_value=0)
old_dependents = st.number_input('Old Dependents', min_value=0)
young_dependents = st.number_input('Young Dependents', min_value=0)
occupants_count = st.number_input('Occupants Count', min_value=1)
house_area = st.number_input('House Area (sq ft)', min_value=100)
loan_tenure = st.number_input('Loan Tenure (months)', min_value=1)
loan_installments = st.number_input('Loan Installments', min_value=1)
loan_amount = st.number_input('Loan Amount', min_value=1000)

#creating user friendly inputs as user will not undestand what does 1&0 mean, F & M etc
sex_mapping={"Male":"M","Female":"F"}
type_of_house_mapping={"Type 1":"T1","Type 2":"T2","Rented":"R"}  
home_ownership_mapping={"Owned":1,"Rented":0}

sex = st.selectbox('Sex',sex_mapping.keys())
type_of_house = st.selectbox('Type of House',type_of_house_mapping.keys())
home_ownership = st.selectbox('Home Ownership',home_ownership_mapping.keys())

sex_encode=sex_mapping[sex]
type_of_house_encode=type_of_house_mapping[type_of_house]
home_ownership_encode=home_ownership_mapping[home_ownership]

# Prepare input data
data = {
    "age": age,
    "annual_income": annual_income,
    "monthly_expenses": monthly_expenses,
    "old_dependents": old_dependents,
    "young_dependents": young_dependents,
    "occupants_count": occupants_count,
    "house_area": house_area,
    "loan_tenure": loan_tenure,
    "loan_installments": loan_installments,
    "loan_amount": loan_amount,
    "sex": sex_encode,
    "type_of_house": type_of_house_encode,
    "home_ownership": home_ownership_encode
}

if st.button('Predict Loan Approval'):
    try:
        # Call API for prediction
        responce=requests.post("http://localhost:8000/predict",json=data)
        if responce.status_code==200:
            prediction=responce.json()["prediction"]
            st.success(f'Loan Approval Status: {"Approved" if prediction == 1 else "Rejected"}')
        else:
            st.error('Error: Unable to fetch prediction from API')
    except Exception as e:
        st.error(f'Exception occurred: {e}')
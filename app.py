import streamlit as st 
import main

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Loan Approval"])

if page == "Home":
    st.title("Welcome to Loan Approval Prediction App")
    st.write("""
        This application helps predict loan approvals for individuals in rural areas using Machine Learning.
        \n**Instructions:**
        - Navigate to the **Predict Loan Approval** tab.
        - Enter the applicant details.
        - Click on **Predict Loan Approval** to see the result.
    """)
elif page == "Predict Loan Approval":
    main.run_app()  # Call the function from main.py
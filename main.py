from src.component.data_ingestion import DataInjestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer
from src.pipeline.predict_pipeline import CustomeData,PredictPipeline

import streamlit as st

def run_app():
    # Data Ingestion
    """ingestion = DataInjestion()
    train_data_path, test_data_path = ingestion.initiate_data_injestion()
    transformation=DataTransformation()
    train_arr,test_arr,preprocessed_file=transformation.initiate_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    score=model_trainer.initiate_model_trainer(train_arr,test_arr)
    print(score)"""

    st.title("Loan Approval Prediction for Rural India")
    st.write("Enter the applicant details below to predict loan approval.")
    # Input Variables
    age = st.number_input("Enter Age", min_value=0, step=1)
    annual_income = st.number_input("Enter Annual Income", min_value=0, step=500)
    monthly_expenses = st.number_input("Enter Monthly Expenses", min_value=0, step=500)
    old_dependents = st.number_input("Enter Number of Old Dependents", min_value=0, step=1)
    young_dependents = st.number_input("Enter Number of Young Dependents", min_value=0, step=1)
    occupants_count = st.number_input("Enter Number of Occupants in the House", min_value=0, step=1)
    house_area = st.number_input("Enter House Area (sq. ft.)", min_value=0, step=10)
    loan_tenure = st.number_input("Enter Loan Tenure (in years)", min_value=0, step=1)
    loan_installments = st.number_input("Enter Loan Installments", min_value=0, step=1)
    loan_amount = st.number_input("Enter Loan Amount", min_value=0, step=500)

    # Categorical Variables
    sex = st.selectbox("Select Sex", options=["M", "F","TG"])
    type_of_house = st.selectbox("Select Type of House", options=["T2", "T1", "R"])
    home_ownership = st.selectbox("Select Home Ownership", options=[1,0])

    if st.button("Predict Loan Approval"):
        input_data=CustomeData(
            age, annual_income, monthly_expenses, old_dependents,
            young_dependents, occupants_count, house_area, loan_tenure,
            loan_installments, loan_amount, sex, type_of_house, home_ownership
        )

        input_df=input_data.get_data_as_dataframe()

        pipeline=PredictPipeline()
        prediction=pipeline.predict(input_df)

        if prediction[0]==0:
            st.write("Medium Risk")
        elif prediction[0]==2:
            st.warning("Low Risk")
        else:
            st.write("High Risk")





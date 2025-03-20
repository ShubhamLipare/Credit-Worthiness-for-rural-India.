import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.util import load_object


class CustomeData:
    def __init__(self,
        age, annual_income, monthly_expenses, old_dependents,
        young_dependents, occupants_count, house_area, loan_tenure, 
        loan_installments, loan_amount, sex, type_of_house, home_ownership):

        self.age = age
        self.annual_income = annual_income
        self.monthly_expenses = monthly_expenses
        self.old_dependents = old_dependents
        self.young_dependents = young_dependents
        self.occupants_count = occupants_count
        self.house_area = house_area
        self.loan_tenure = loan_tenure
        self.loan_installments = loan_installments
        self.loan_amount = loan_amount
        self.sex = sex
        self.type_of_house = type_of_house
        self.home_ownership = home_ownership

    def get_data_as_dataframe(self):

        try:
            custome_input_data_dict={
            "age": [self.age],
            "annual_income": [self.annual_income],
            "monthly_expenses": [self.monthly_expenses],
            "old_dependents": [self.old_dependents],
            "young_dependents": [self.young_dependents],
            "occupants_count": [self.occupants_count],
            "house_area": [self.house_area],
            "loan_tenure": [self.loan_tenure],
            "loan_installments": [self.loan_installments],
            "loan_amount": [self.loan_amount],
            "sex": [self.sex],
            "type_of_house": [self.type_of_house],
            "home_ownership": [self.home_ownership],
            }

            return pd.DataFrame(custome_input_data_dict)

        except Exception as e:
            raise CustomException(e,sys)

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):

        try:

            model_path=os.path.join("artifacts","model.pkl")
            preprocessing_path=os.path.join("artifacts","preprocessor.pkl")

            model=load_object(model_path)
            preprocessor=load_object(preprocessing_path)
            logging.info("Model and Preprocessor has been loaded sucessfully")

            processed_features=preprocessor.transform(features)
            pred=model.predict(processed_features)

            return pred

        except Exception as e:
            raise CustomException(e,sys)


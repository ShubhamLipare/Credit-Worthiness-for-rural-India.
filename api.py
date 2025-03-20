from fastapi import FastAPI,HTTPException
import pandas as pd
from pydantic import BaseModel
from src.pipeline.predict_pipeline import CustomeData,PredictPipeline

app=FastAPI()

class InputData(BaseModel):
    age: int
    annual_income: float
    monthly_expenses: float
    old_dependents: int
    young_dependents: int
    occupants_count: int
    house_area: float
    loan_tenure: int
    loan_installments: int
    loan_amount: float
    sex: str
    type_of_house: str
    home_ownership: int

@app.post("/predict")
def predict(data:InputData):
    try:
        # Convert input to DataFrame
        input_data = CustomeData(
            age=data.age,
            annual_income=data.annual_income,
            monthly_expenses=data.monthly_expenses,
            old_dependents=data.old_dependents,
            young_dependents=data.young_dependents,
            occupants_count=data.occupants_count,
            house_area=data.house_area,
            loan_tenure=data.loan_tenure,
            loan_installments=data.loan_installments,
            loan_amount=data.loan_amount,
            sex=data.sex,
            type_of_house=data.type_of_house,
            home_ownership=data.home_ownership
        ).get_data_as_dataframe()

        pipeline=PredictPipeline()
        prediction=pipeline.predict(input_data)[0]

        return {"prediction":int(prediction)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
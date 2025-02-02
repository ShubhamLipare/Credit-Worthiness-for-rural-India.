### Data Ingestion (data_ingestion.py)
```python
import os
import pandas as pd

def ingest_data():
    df = pd.read_csv("data/source_data.csv")
    os.makedirs("artifacts", exist_ok=True)
    df.to_csv("artifacts/raw_data.csv", index=False)
```

### Data Transformation (data_transformation.py)
```python
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

def transform_data():
    df = pd.read_csv("artifacts/raw_data.csv")
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=["number"]))
    pickle.dump(scaler, open("artifacts/preprocessor.pkl", "wb"))
    pd.DataFrame(df_scaled).to_csv("artifacts/transformed_data.csv", index=False)
```

### Model Training (model_trainer.py)
```python
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model():
    df = pd.read_csv("artifacts/transformed_data.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    
    mlflow.set_experiment("loan_approval")
    with mlflow.start_run():
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        pickle.dump(model, open("artifacts/model.pkl", "wb"))
```

### Model Serving (api.py)
```python
from fastapi import FastAPI
import pickle
import pandas as pd

def load_model():
    return pickle.load(open("artifacts/model.pkl", "rb"))

def load_preprocessor():
    return pickle.load(open("artifacts/preprocessor.pkl", "rb"))

app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    preprocessor = load_preprocessor()
    model = load_model()
    transformed_data = preprocessor.transform(df)
    prediction = model.predict(transformed_data)
    return {"prediction": prediction.tolist()}
```

### Model Monitoring (drift_monitoring.py)
```python
import evidently
from evidently.test_suite import TestSuite
from evidently.tests import TestDataDrift
import pandas as pd

def monitor_drift():
    reference_data = pd.read_csv("artifacts/transformed_data.csv")
    new_data = pd.read_csv("data/new_data.csv")
    test_suite = TestSuite(tests=[TestDataDrift()])
    test_suite.run(reference_data=reference_data, current_data=new_data)
    test_suite.save_html("artifacts/drift_report.html")
```

### Streamlit UI (app.py)
```python
import streamlit as st
import requests

def main():
    st.title("Loan Approval Prediction")
    st.write("Enter the required details for loan approval prediction.")
    
    input_data = {}
    input_data["feature1"] = st.number_input("Feature 1")
    input_data["feature2"] = st.number_input("Feature 2")
    # Add more features as needed
    
    if st.button("Predict"):
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
        st.write("Prediction:", response.json()["prediction"])

if __name__ == "__main__":
    main()
```

### Main Processing (main.py)
```python
import pickle
import pandas as pd

def load_model():
    return pickle.load(open("artifacts/model.pkl", "rb"))

def load_preprocessor():
    return pickle.load(open("artifacts/preprocessor.pkl", "rb"))

def predict(input_data):
    df = pd.DataFrame([input_data])
    preprocessor = load_preprocessor()
    model = load_model()
    transformed_data = preprocessor.transform(df)
    return model.predict(transformed_data)[0]
```

### GitHub Actions Workflow (.github/workflows/ml_pipeline.yml)
```yaml
name: ML Pipeline

on:
  schedule:
    - cron: "0 0 * * *"  # Runs daily
  workflow_dispatch:

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v3
      
      - name: Set Up Python
        uses: actions/setup-python@v3
        with:
          python-version: "3.9"
      
      - name: Install Dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Data Ingestion
        run: python data_ingestion.py
      
      - name: Data Transformation
        run: python data_transformation.py
      
      - name: Train Model
        run: python model_trainer.py
      
      - name: Monitor Data Drift
        run: python drift_monitoring.py
```

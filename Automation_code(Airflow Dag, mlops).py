### Data Ingestion (data_ingestion.py)
```python
import os
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def ingest_data():
    df = pd.read_csv("data/source_data.csv")
    os.makedirs("artifacts", exist_ok=True)
    df.to_csv("artifacts/raw_data.csv", index=False)

def refresh_data():
    ingest_data()

define_dag = DAG(
    dag_id="data_ingestion_pipeline",
    schedule_interval="@daily",
    start_date=datetime(2025, 2, 1),
    catchup=False
)

ingest_task = PythonOperator(
    task_id="ingest_data",
    python_callable=refresh_data,
    dag=define_dag
)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model():
    df = pd.read_csv("artifacts/transformed_data.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
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

### Airflow DAG for End-to-End Pipeline (ml_pipeline.py)
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from data_ingestion import ingest_data
from data_transformation import transform_data
from model_trainer import train_model
from drift_monitoring import monitor_drift

define_dag = DAG(
    dag_id="ml_pipeline",
    schedule_interval="@daily",
    start_date=datetime(2025, 2, 1),
    catchup=False
)

ingest_task = PythonOperator(task_id="ingest_data", python_callable=ingest_data, dag=define_dag)
transform_task = PythonOperator(task_id="transform_data", python_callable=transform_data, dag=define_dag)
train_task = PythonOperator(task_id="train_model", python_callable=train_model, dag=define_dag)
drift_task = PythonOperator(task_id="monitor_drift", python_callable=monitor_drift, dag=define_dag)

ingest_task >> transform_task >> train_task >> drift_task
```

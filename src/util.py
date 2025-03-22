import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,recall_score

def save_object(filepath,obj):
    try:
        dir_path=os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)

        with open(filepath,"wb") as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(filepath):
    try:
        with open(filepath,"rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train,x_test,y_train,y_test,models,params):
    try:
        logging.info("Enterted into evaluation util")
        train_report={}
        test_report={}
        mlflow.set_experiment("Model Evaluation")
        mlflow.set_tracking_uri("https://credit-worthiness-for-rural-india.onrender.com")

        #Hyperparameter tuning is not used here as data size is high 
        #not able to train model

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model=list(models.values())[i]
            param=params[list(models.keys())[i]]

            with mlflow.start_run(run_name=model_name):
                for key,value in param.items():
                    mlflow.log_param(key,value)

                model.fit(x_train,y_train)
                logging.info(f"{model_name} fitting is completed")

                y_train_pred=model.predict(x_train)
                y_test_pred=model.predict(x_test)
                logging.info("Prediction is completed")

                train_model_score=recall_score(y_train,y_train_pred,average='macro')
                test_model_score=recall_score(y_test,y_test_pred,average='macro')
                logging.info("Recall score is calculated")

                logging.info("mlflow logging metrics")
                mlflow.log_metric("Training accuracy",train_model_score)
                mlflow.log_metric("Testing accuracy",test_model_score)
                mlflow.sklearn.log_model(model,model_name)
            
                logging.info(f"generating models report for {model_name}")
                train_report[list(models.keys())[i]]=train_model_score
                test_report[list(models.keys())[i]]=test_model_score

        
        return train_report,test_report
        
    except Exception as e:
        raise CustomException(e,sys)
    
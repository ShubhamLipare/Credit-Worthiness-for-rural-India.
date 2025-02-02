import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score

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
        for i in range(len(list(models))):
            model=list(models.values())[i]
            param=params[list(models.keys())[i]]

            #gs=GridSearchCV(model,param,cv=3)
            #gs.fit(x_train,y_train)
            model.fit(x_train,y_train)
            logging.info("Model fitting is completed")

            #model.set_params(**gs.best_params_)
            #model.fit(x_train,y_train)

            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)
            logging.info("Prediction is completed")

            train_model_score=accuracy_score(y_train_pred,y_train)
            test_model_score=accuracy_score(y_test_pred,y_test)
            
            logging.info("generating models report")
            train_report[list(models.keys())[i]]=train_model_score
            test_report[list(models.keys())[i]]=test_model_score
        
        return train_report,test_report
        
    except Exception as e:
        raise CustomException(e,sys)
    
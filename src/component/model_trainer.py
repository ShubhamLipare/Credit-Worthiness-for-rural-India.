import os
import sys
import pandas as pd

from sklearn.ensemble import (AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from dataclasses import dataclass
from src.logger import logging
from src.util import evaluate_model,save_object
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
        trained_model_file_path=os.path.join("artifacts","model.pkl")
        model_train_report_path=os.path.join("artifacts","model_train_report.csv")
        model_test_report_path=os.path.join("artifacts","model_test_report.csv")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Model  training has begun")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(),
            "AdaBoost Classifier": AdaBoostClassifier(),
            }

            params = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy', 'log_loss'],
                # 'splitter': ['best', 'random'],
                # 'max_features': ['sqrt', 'log2'],
            },
            "Random Forest": {
                # 'criterion': ['gini', 'entropy', 'log_loss'],
                # 'max_features': ['sqrt', 'log2', None],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Gradient Boosting": {
                # 'loss': ['log_loss', 'deviance', 'exponential'],
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                # 'criterion': ['friedman_mse'],
                # 'max_features': ['auto', 'sqrt', 'log2'],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Logistic Regression": {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs', 'saga']
            },
            "AdaBoost Classifier": {
                'learning_rate': [0.1, 0.01, 0.5, 0.001],
                # 'algorithm': ['SAMME', 'SAMME.R'],
                'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_train_report,model_test_report=evaluate_model(x_train,x_test,y_train,y_test,models,params)
            logging.info("Saving train and test report to artifacts folder")
            pd.DataFrame.from_dict(model_train_report, orient="index", columns=["accuracy"]).to_csv(self.model_trainer_config.model_train_report_path)
            pd.DataFrame.from_dict(model_test_report, orient="index", columns=["accuracy"]).to_csv(self.model_trainer_config.model_test_report_path)

            best_model_score=max(sorted(model_test_report.values()))
            best_model_name=list(model_test_report.keys())[
                list(model_test_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            
            logging.info("evaluation completed")
            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found for both training and test data name:{best_model_name}")
            logging.info(f"{best_model_score}")

            save_object(
                filepath=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
            accuracy=accuracy_score(predicted,y_test)

            return accuracy

            

        except Exception as e:
            CustomException(e,sys)

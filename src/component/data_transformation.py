import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from sklearn.base import BaseEstimator,TransformerMixin

from src.logger import logging
from src.exception import CustomException
from src.util import save_object
from src.component.clustering import Cluster

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join("artifacts","preprocessor.pkl")
    cluster_df_path:str=os.path.join("artifacts","cluster_df.csv")
    train_arr_csv:str=os.path.join("artifacts","train_arr.csv")
    test_arr_csv:str=os.path.join("artifacts","test_arr.csv")
    train_arr_with_cluster:str=os.path.join("artifacts","train_arr_with_cluster.csv")
    test_arr_with_cluster:str=os.path.join("artifacts","test_arr_with_cluster.csv")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    # Custom Transformer: Drop Unnecessary Columns
    class ColumnDropper(BaseEstimator,TransformerMixin):
        def __init__(self,columns_to_drop):
            self.columns_to_drop=columns_to_drop

        def fit(self,X,y=None):
            return self
        
        def transform(self,X):
            return X.drop(self.columns_to_drop, errors="ignore")
 
    # Custom Transformer: Data Type Conversion
    class DataTypeConvertor(BaseEstimator,TransformerMixin):
        def __init__(self,column_datatype_mapping):
            self.column_datatype_mapping=column_datatype_mapping

        def fit(self,X,y=None):
            return self
        
        def transform(self,X):
            for col,dtype in self.column_datatype_mapping.items():
                if col in X.columns:
                    X[col]=X[col].astype(dtype)
            return X
        
    # Custom Transformer: Outlier Detection and Treatment
    class OutlierTreatment(BaseEstimator,TransformerMixin):
        def __init__(self,method="iqr",threshold=1.5):
            self.method=method
            self.threshold=threshold
        
        def fit(self,X,y=None):
            return self
        
        def transform(self,X):
            if self.method=="iqr":
                for column in X.select_dtypes(include=["int64","float64"]).columns:
                    Q1 = X[column].quantile(0.25)
                    Q3 = X[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.threshold * IQR
                    upper_bound = Q3 + self.threshold * IQR
                    X[column] = X[column].clip(lower=lower_bound, upper=upper_bound)
            return X
    def get_data_transformer_object(self):
        try:
            columns_to_drop = ["Id","social_class","primary_business","secondary_business","sanitary_availability","water_availabity", 
                               "type_of_house","loan_purpose"]

            # Data type conversions
            column_dtype_mapping = {
                "age": "int64",
                "home_ownership":"int64",
                "occupants_count":"int64"
            }

            # Numerical and categorical columns
            numerical_columns = [
                'age', 'annual_income', 'monthly_expenses', 'old_dependents', 'young_dependents', 'occupants_count', 'house_area', 'loan_tenure', 'loan_installments','loan_amount'
            ]
            categorical_columns = ['sex', 'type_of_house', 'home_ownership']

            num_pipeline=Pipeline(steps=[
                ("drop_columns",self.ColumnDropper(columns_to_drop)),
                 ("convert_dtype",self.DataTypeConvertor(column_dtype_mapping)),
                 ("treat_outliers",self.OutlierTreatment(method="iqr")),
                 ("impute_missing_values",SimpleImputer(strategy="median")),
                 ("scaler",StandardScaler())
            ])

            cat_pipeline=Pipeline(steps=[
                ("impute_missing_values",SimpleImputer(strategy="most_frequent")),
                ("encoding",OneHotEncoder())
            ])

            preprocessor=ColumnTransformer(
                transformers=[
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            logging.info("Preprocessing pipeline created.")

            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read data from artifacts")

            preprocessor_obj=self.get_data_transformer_object()
            logging.info("preprocessor object created")

            train_arr_preprocessed=preprocessor_obj.fit_transform(train_df)
            test_arr_preprocessed=preprocessor_obj.transform(test_df)
            logging.info("Preprocessing has been done on training and testing data")
            pd.DataFrame(train_arr_preprocessed).to_csv(self.data_transformation_config.train_arr_csv)
            pd.DataFrame(test_arr_preprocessed).to_csv(self.data_transformation_config.test_arr_csv)
            logging.info("Preprocessed training and testing data is stored in artifacts folder")

            cluster_obj=Cluster()
            train_arr_with_cluster,test_arr_with_cluster=cluster_obj.make_cluster(train_arr=train_arr_preprocessed,test_arr=test_arr_preprocessed,n_cluster=3)
            pd.DataFrame(train_arr_with_cluster).to_csv(self.data_transformation_config.train_arr_with_cluster)
            pd.DataFrame(test_arr_with_cluster).to_csv(self.data_transformation_config.test_arr_with_cluster)
            logging.info("clustering is completed and saved data")

            save_object(
                filepath=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr_with_cluster,test_arr_with_cluster,
                self.data_transformation_config.preprocessor_obj_file_path
    
            )
        except Exception as e:
            raise CustomException(e,sys)
        


                

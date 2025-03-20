import os
import sys
from dataclasses import dataclass
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    data_path:str=os.path.join("artifacts","raw_data.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")


class DataInjestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_injestion(self):
        logging.info("Entere data injestion method or object")

        try:

            df=pd.read_csv(r"Data\trainingData .csv")
            logging.info("Read the specified dataset")

            ##creating artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.data_path),exist_ok=True)
            logging.info("artifacts folder created")

            #storing raw data as csv
            df.to_csv(self.ingestion_config.data_path,index=False,header=True)
            logging.info("data is stored")

            train_data,test_data=train_test_split(df,test_size=0.3,random_state=42)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("train test data is stored")



            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)

from src.component.data_ingestion import DataInjestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer
from src.pipeline.predict_pipeline import CustomeData,PredictPipeline
from src.exception import CustomException
from src.logger import logging
import sys

def run_app():
    try:
        # Data Ingestion
        ingestion = DataInjestion()
        train_data_path, test_data_path = ingestion.initiate_data_injestion()
        logging.info("Data injestion completed.")

        transformation=DataTransformation()
        train_arr,test_arr,preprocessed_file=transformation.initiate_data_transformation(train_data_path,test_data_path)
        logging.info("Preprocessing is completed.")

        model_trainer=ModelTrainer()
        score=model_trainer.initiate_model_trainer(train_arr,test_arr)
        print(score)

    except Exception as e:
        raise CustomeData(e,sys)

if __name__=="__main__":
    run_app()


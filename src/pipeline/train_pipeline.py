import sys

from src.exception import CustomException
from src.components import (DataIngestion, DataTransformation, ModelTrainer)
from src.logger import logging

class TrainPipeline:
    
    def __init__(self):
        self.data_ingestor = DataIngestion()
        self.data_transformer = DataTransformation()
        self.model_trainer = ModelTrainer()        
    
    def initiate_train_pipeline(self):
        logging.info("Training Pipeline initiated.")
        try:
            train_data_path, test_data_path = self.data_ingestor.initiate_data_ingestion()
            
            X_train, X_test, y_train, y_test = self.data_transformer.initiate_data_transformation(
                train_data_path, test_data_path
            )
            
            self.model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)
            
            logging.info("Training Pipeline completed.")
        except Exception as e:
            raise CustomException(e, sys)
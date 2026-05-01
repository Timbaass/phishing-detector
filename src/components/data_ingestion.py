import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join(BASE_DIR, "data", "raw", "TREC-05.csv")
    train_data_path: str = os.path.join(BASE_DIR, "data", "processed", "train.csv")
    test_data_path: str = os.path.join(BASE_DIR, "data", "processed", "test.csv")

class DataIngestion:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion has started.")

        try:
            df = pd.read_csv(self.data_ingestion_config.raw_data_path,
                            sep=',',  
                            quotechar='"',        
                            on_bad_lines='warn', 
                            engine='python',
                            encoding='latin-1')
            logging.info("Dataset has turned into dataframe from csv.")
            
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok = True)
            
            logging.info("Train test split initiated.")
            
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)
            
            train_set.to_csv(self.data_ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index = False, header = True)
            
            logging.info("Data Ingestion has been completed.")
            
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as ex:
            raise CustomException(ex, sys) 
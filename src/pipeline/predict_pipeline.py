import sys
import os

from src.config.paths import ARTIFACTS_DIR
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

from dataclasses import dataclass

@dataclass
class PredictPipelineConfig:
    model_file_path: str = os.path.join(ARTIFACTS_DIR, "model.pkl")
    preprocesser_file_path: str = os.path.join(ARTIFACTS_DIR, "preprocesser.pkl")
    
class PredictPipeline:
    
    def __init__(self):
        self.predict_pipeline_config = PredictPipelineConfig()
        
    def predict(self, input):
        logging.info("Prediction Pipeline initiated.")
        try:
            model = load_object(self.predict_pipeline_config.model_file_path)
            preprocesser = load_object(self.predict_pipeline_config.preprocesser_file_path)
            
            scaled_input = preprocesser.transform(input)
            prediction = model.predict(scaled_input)
            
            logging.info("Prediction Pipeline completed.")
            
            return prediction
        except Exception as e:
            raise CustomException(e, sys)
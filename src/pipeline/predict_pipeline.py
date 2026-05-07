import sys
import os
from typing import Optional, Tuple

from src.config.paths import ARTIFACTS_DIR
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

from dataclasses import dataclass
import pandas as pd

@dataclass
class PredictPipelineConfig:
    model_file_path: str = os.path.join(ARTIFACTS_DIR, "model.pkl")
    preprocesser_file_path: str = os.path.join(ARTIFACTS_DIR, "preprocesser.pkl")
    
class PredictPipeline:

    _model_cache: Optional[object] = None
    _preprocessor_cache: Optional[object] = None

    def __init__(self):
        self.predict_pipeline_config = PredictPipelineConfig()

    def predict(self, input_data):
        logging.info("Prediction Pipeline initiated.")
        try:
            model, preprocesser = self._load_artifacts()
            scaled_input = preprocesser.transform(input_data)
            prediction = model.predict(scaled_input)
            proba = self.get_predict_proba(model, scaled_input)
            
            logging.info("Prediction Pipeline completed.")
            
            return prediction, proba
        except Exception as e:
            raise CustomException(e, sys)

    def _load_artifacts(self) -> Tuple[object, object]:
        if self.__class__._model_cache is not None and self.__class__._preprocessor_cache is not None:
            return self.__class__._model_cache, self.__class__._preprocessor_cache

        model_path = self.predict_pipeline_config.model_file_path
        preprocessor_path = self.predict_pipeline_config.preprocesser_file_path

        if not os.path.exists(model_path):
            raise CustomException(f"Model file not found: {model_path}", sys)
        if not os.path.exists(preprocessor_path):
            raise CustomException(f"Preprocessor file not found: {preprocessor_path}", sys)

        self.__class__._model_cache = load_object(model_path)
        self.__class__._preprocessor_cache = load_object(preprocessor_path)

        return self.__class__._model_cache, self.__class__._preprocessor_cache
        
    def get_predict_proba(self, model: object, scaled_input):
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(scaled_input)
                if proba is None:
                    return None

                if proba.ndim == 2 and proba.shape[1] > 1:
                    return proba[:, 1]

                return proba
            
            return None
        except Exception as e:
            raise CustomException(e, sys)

@dataclass
class CustomData:
    sender: str
    receiver: str
    date: str
    subject: str
    body: str

    def get_data_as_dataframe(self):
        return pd.DataFrame({
            "sender": [self.sender],
            "receiver": [self.receiver],
            "date": [self.date],
            "subject": [self.subject],
            "body": [self.body],
        })

import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.config.paths import ARTIFACTS_DIR
from src.utils import save_object, evaluate_models

from dataclasses import dataclass

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

@dataclass
class ModelTrainerConfig:
    model_obj_file_path: str = os.path.join(ARTIFACTS_DIR, "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Model training initiated.")
            models = {
                "Random Forest Classifier Model": RandomForestClassifier(),
                "LinerSVC Model": LinearSVC(),
                "Logistic Regressor Model": LogisticRegression(max_iter=5000, solver="liblinear"),
                "XGBClassifier Model": XGBClassifier() 
            }
            
            model_report, trained_models = evaluate_models(models, X_train, y_train, X_test, y_test)
            
            logging.info("Model training completed.")
            
            best_model_name = max(model_report, key = lambda model: model_report[model]["F1"])
            
            best_model = trained_models[best_model_name]
            
            if model_report[best_model_name]["F1"] < 0.6:
                raise CustomException("Could not find any best model.")
            
            save_object(
                file_path = self.model_trainer_config.model_obj_file_path,
                obj = best_model
            )
            
            return model_report
        except Exception as e:
            raise CustomException(e, sys)
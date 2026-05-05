import dill, sys, os
from src.exception import CustomException

from sklearn import metrics

def save_object(file_path: str, obj):
    """Save a object to disk."""
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok = True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str):
    """Load a object from disk."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(models: dict, X_train, y_train, X_test, y_test):

    try:

        model_report = {}
        trained_models = {}

        for model_name, model_obj in models.items():

            model_obj.fit(X_train, y_train)

            trained_models[model_name] = model_obj

            preds = model_obj.predict(X_test)

            model_report[model_name] = {
                "Accuracy": metrics.accuracy_score(y_test, preds),
                "Precision": metrics.precision_score(y_test, preds),
                "Recall": metrics.recall_score(y_test, preds),
                "F1": metrics.f1_score(y_test, preds)
            }

        return model_report, trained_models

    except Exception as e:
        raise CustomException(e, sys)
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.components.feature_extractor import FeatureExtractor, NUMERIC_FEATURES, TEXT_COLUMNS
from src.config.paths import ARTIFACTS_DIR
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocesser_obj_file_path: str = os.path.join(ARTIFACTS_DIR, "preprocesser.pkl")


class DataTransformation:
    """Prepare and persist preprocessing pipeline for training and inference."""

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) -> Pipeline:
        """Build a reusable sklearn Pipeline for text and numeric features."""
        try:
            tfidf = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("tfidf", tfidf, "text_clean"),
                    ("numeric", "passthrough", NUMERIC_FEATURES),
                ],
                remainder="drop",
            )

            return Pipeline(
                steps=[
                    ("feature_engineering", FeatureExtractor()),
                    ("preprocessor", preprocessor),
                ]
            )
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """Fit the preprocessing pipeline and transform train/test datasets."""
        logging.info("Data transformation initiated.")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            if "label" not in train_df.columns or "label" not in test_df.columns:
                raise ValueError("Both train and test data must include a 'label' column.")

            train_df = train_df[train_df["label"].isin([0, 1, "0", "1"])].copy()
            test_df = test_df[test_df["label"].isin([0, 1, "0", "1"])].copy()

            X_train = train_df.drop(columns=["label"])
            y_train = train_df["label"].astype(int).to_numpy()
            X_test = test_df.drop(columns=["label"])
            y_test = test_df["label"].astype(int).to_numpy()

            preprocessor = self.get_data_transformer_object()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            save_object(
                self.data_transformation_config.preprocesser_obj_file_path,
                preprocessor,
            )

            logging.info("Data transformation completed.")
            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.data_transformation_config.preprocesser_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
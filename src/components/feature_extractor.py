import re
import sys
from email.utils import parsedate_to_datetime
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from src.components.data_cleaner import DataCleaner
from src.config.constants import (
    HTML_PATTERN,
    NUMERIC_FEATURES,
    TEXT_COLUMNS,
    URGENT_KEYWORDS,
    URL_PATTERN,
)
from src.exception import CustomException

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Generate cleaned text and numeric features from email data."""

    def __init__(self, cleaner: Optional[DataCleaner] = None):
        self.cleaner = cleaner or DataCleaner()

    def fit(self, X, y=None) -> "FeatureExtractor":
        return self

    def transform(self, X) -> pd.DataFrame:
        try:
            df = self._normalize_input(X)
            df = self._fill_missing_text(df)
            numeric_df = self._build_numeric_features(df)
            text_df = self._build_clean_text(df)
            return pd.concat([text_df, numeric_df], axis=1)
        except Exception as e:
            raise CustomException(e, sys)

    def _normalize_input(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, pd.Series):
            df = pd.DataFrame({"subject": "", "body": X})
        else:
            df = pd.DataFrame({"subject": "", "body": list(X)})

        missing = [col for col in TEXT_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df

    def _fill_missing_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df["subject"] = df["subject"].fillna("").astype(str)
        df["body"] = df["body"].fillna("").astype(str)
        return df

    def _build_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        raw_text = df["subject"] + " " + df["body"]

        features = pd.DataFrame(
            {
                "url_count": df["body"].str.count(URL_PATTERN),
                "contains_urgent_word": raw_text.apply(
                    self._contains_urgent_keyword
                ),
                "body_len": df["body"].str.len(),
                "is_html": df["body"].str.contains(HTML_PATTERN, regex=True).astype(
                    int
                ),
                "exclamation_count": df["body"].str.count("!") + df["subject"].str.count("!"),
                "time_bucket": self._bucketize_time(df),
            }
        )

        return features.astype("float32")

    def _build_clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = self.cleaner.transform(df[TEXT_COLUMNS])
        text_clean = (cleaned["subject"] + " " + cleaned["body"]).str.strip()
        return pd.DataFrame({"text_clean": text_clean})

    def _bucketize_time(self, df: pd.DataFrame) -> pd.Series:
        if "date" not in df.columns:
            return pd.Series(-1, index=df.index)
        hours = df["date"].apply(self._extract_hour)
        buckets = np.select(
            [
                hours.between(5, 11),
                hours.between(12, 16),
                hours.between(17, 21),
            ],
            [0, 1, 2],
            default=-1,
        )
        return pd.Series(buckets, index=df.index)

    @staticmethod
    def _extract_hour(value) -> float:
        if pd.isna(value):
            return np.nan

        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            return parsed.hour

        try:
            parsed = parsedate_to_datetime(str(value))
        except Exception:
            return np.nan

        if parsed is None:
            return np.nan

        return parsed.hour

    @staticmethod
    def _contains_urgent_keyword(text: str) -> int:
        text = str(text).lower()
        for keyword in URGENT_KEYWORDS:
            if " " in keyword:
                if keyword in text:
                    return 1
            elif re.search(rf"\\b{re.escape(keyword)}\\b", text):
                return 1
        return 0

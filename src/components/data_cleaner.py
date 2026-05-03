import pandas as pd
import sys
import re

from src.config.constants import STOP_WORDS
from src.exception import CustomException

from bs4 import BeautifulSoup
import contractions

from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

class DataCleaner(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        try:
            if "<" in text and ">" in text:
                text = BeautifulSoup(text, "html.parser").get_text() #HTML elemntlerinden kurtulur.
                
            text = contractions.fix(text) # İngilizce kelimeleri ayırır.
            
            text = str(text).lower()
            text = re.sub(r'[-/]', ' ', text)
            text = re.sub(r"[^\w\s]", " ", text) # noktalama işaretleri yerine boşluk koyar.
            text = re.sub(r'\s+', ' ', text).strip() # Birden fazla boşluğu teke indirir.
            tokens = [
                self.lemmatizer.lemmatize(token)
                for token in text.split()
                if token not in STOP_WORDS and token.isalpha()
            ]
            return " ".join(tokens)
        except Exception as e:
            raise CustomException(e, sys)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")

            return X.apply(lambda col: col.apply(lambda x: self.clean_text(x)))

        except Exception as e:
            raise CustomException(e, sys)
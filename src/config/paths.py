import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
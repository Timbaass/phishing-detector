from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

URGENT_KEYWORDS = {
    "urgent",
    "immediately",
    "asap",
    "action required",
    "verify",
    "suspended",
    "deadline",
    "final notice",
    "account",
    "payment",
    "update",
}

NUMERIC_FEATURES = [
    "url_count",
    "contains_urgent_word",
    "body_len",
    "is_html",
    "exclamation_count",
    "time_bucket",
]

TEXT_COLUMNS = ["subject", "body"]

STOP_WORDS = set(ENGLISH_STOP_WORDS)

URL_PATTERN = re.compile(r"(https?://\\S+|www\\.\\S+)", flags=re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]+>")
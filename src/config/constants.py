from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re, string

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

STOP_WORDS = set(ENGLISH_STOP_WORDS)

URL_PATTERN = re.compile(r"(https?://\\S+|www\\.\\S+)", flags=re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]+>")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

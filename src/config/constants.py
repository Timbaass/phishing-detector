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

HREF_PATTERN = re.compile(
    r"""href\s*=\s*(?:"(?P<double>[^"]*)"|'(?P<single>[^']*)'|(?P<bare>[^\s>]+))""",
    flags=re.IGNORECASE,
)
PLAIN_URL_PATTERN = re.compile(
    r"""(?<![@\w.-])(?P<url>(?:https?://[^\s<>"']+|www\.[^\s<>"']+|(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}(?:[/:?#][^\s<>"']*)?))""",
    flags=re.IGNORECASE,
)
BARE_DOMAIN_PATTERN = re.compile(
    r"""^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}(?:[/:?#].*)?$""",
    flags=re.IGNORECASE,
)
IGNORED_SCHEMES = ("mailto:", "tel:", "javascript:")
WRAPPER_CHARS = "<>()[]{}\"'"
TRAILING_PUNCTUATION = ".,)]}\"';"

SUSPICIOUS_KEYWORDS = {
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "bank",
    "password",
    "wallet",
    "payment",
    "confirm",
}

URL_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
}
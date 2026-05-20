"""Utilities for extracting normalized URLs from email text or HTML."""

from html import unescape
import re

from src.config.constants import (HREF_PATTERN, PLAIN_URL_PATTERN, WRAPPER_CHARS, TRAILING_PUNCTUATION, IGNORED_SCHEMES, BARE_DOMAIN_PATTERN)

def extract_urls(text: str) -> list[str]:
    """Return normalized, deduplicated URLs found in plain text or HTML."""
    if text is None:
        return []

    raw_text = str(text)
    if not raw_text.strip():
        return []

    urls = []
    urls.extend(_extract_href_urls(raw_text))
    urls.extend(_extract_plain_urls(raw_text))
    return _dedupe_keep_order(urls)


def _extract_href_urls(text: str) -> list[str]:
    urls = []
    for match in HREF_PATTERN.finditer(text):
        raw_url = match.group("double") or match.group("single") or match.group("bare")
        normalized = _normalize_url(raw_url)
        if normalized:
            urls.append(normalized)
    return urls


def _extract_plain_urls(text: str) -> list[str]:
    urls = []
    for match in PLAIN_URL_PATTERN.finditer(text):
        normalized = _normalize_url(match.group("url"))
        if normalized:
            urls.append(normalized)
    return urls


def _normalize_url(url: str) -> str | None:
    if url is None:
        return None

    normalized = unescape(str(url)).strip().strip(WRAPPER_CHARS)
    normalized = normalized.rstrip(TRAILING_PUNCTUATION)
    if not normalized:
        return None

    lowered = normalized.lower()
    if lowered.startswith(IGNORED_SCHEMES):
        return None

    if lowered.startswith("www.") or _looks_like_bare_domain(normalized):
        normalized = f"http://{normalized}"

    return normalized


def _looks_like_bare_domain(value: str) -> bool:
    return bool(BARE_DOMAIN_PATTERN.match(value))


def _dedupe_keep_order(urls: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


__all__ = ["extract_urls"]

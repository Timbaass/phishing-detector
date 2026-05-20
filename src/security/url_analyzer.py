from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
import os, sys, time, requests

from src.exception import CustomException

load_dotenv()

SUSPICIOUS_WEIGHT = 0.5

class UrlAnalyzer:
    
    def __init__(self):
        self.API_KEY = os.getenv("VT_API_KEY")
        self.URL = os.getenv("VT_URL")
        self.headers = {
            "accept": "application/json",
            "content-type": "application/x-www-form-urlencoded",
            "X-Apikey": self.API_KEY
        }
        
    def analyze_url(self, url: str):
        try:
            self._validate_config()
            normalized_url = self._normalize_url(url)
            response = requests.post(
                self.URL,
                headers=self.headers,
                data={"url": normalized_url},
                timeout=15,
            )

            return response
        except Exception as e:
            raise CustomException(e, sys)

    def analyze_url_with_report(
        self,
        url: str,
        poll: bool = True,
        max_attempts: int = 8,
        interval_seconds: float = 1.5,
    ) -> Dict[str, Any]:
        submission = self.analyze_url(url)
        analysis_url = self._extract_analysis_url(submission)

        if not poll:
            return self.get_analysis(analysis_url)

        last_payload: Dict[str, Any] = {}
        for _ in range(max_attempts):
            last_payload = self.get_analysis(analysis_url)
            status = (
                last_payload.get("data", {})
                .get("attributes", {})
                .get("status")
            )
            if status == "completed":
                return last_payload
            time.sleep(interval_seconds)

        return last_payload

    def analyze_urls(self, urls: List[str]):
        return [self.analyze_url(url) for url in urls or []]

    def get_analysis(self, analysis_url: str) -> Dict[str, Any]:
        response = requests.get(analysis_url, headers=self.headers, timeout=15)
        response.raise_for_status()
        return response.json()

    def calculate_vt_score(self, report: Dict[str, Any]) -> Optional[float]:
        stats = (
            report.get("data", {})
            .get("attributes", {})
            .get("stats", {})
        )

        if not isinstance(stats, dict) or not stats:
            return None

        malicious = int(stats.get("malicious", 0) or 0)
        suspicious = int(stats.get("suspicious", 0) or 0)
        harmless = int(stats.get("harmless", 0) or 0)
        undetected = int(stats.get("undetected", 0) or 0)

        total = malicious + suspicious + harmless + undetected
        if total <= 0:
            return None

        detected = malicious + (SUSPICIOUS_WEIGHT * suspicious)
        return round((detected / total) * 100, 2)

    def _normalize_url(self, url: str) -> str:
        if url is None:
            raise ValueError("URL is required")

        normalized = str(url).strip()
        if not normalized:
            raise ValueError("URL is required")

        if normalized.startswith("http://") or normalized.startswith("https://"):
            return normalized

        return f"https://{normalized}"

    def _extract_analysis_url(self, response: requests.Response) -> str:
        response.raise_for_status()
        payload = response.json()
        analysis_url = (
            payload.get("data", {})
            .get("links", {})
            .get("self")
        )

        if analysis_url:
            return analysis_url

        analysis_id = payload.get("data", {}).get("id")
        base_url = self._analysis_base_url()
        if analysis_id and base_url:
            return f"{base_url}/analyses/{analysis_id}"

        raise ValueError("Analysis URL could not be determined from response")

    def _analysis_base_url(self) -> Optional[str]:
        if not self.URL:
            return None
        if "/urls" in self.URL:
            return self.URL.rsplit("/urls", 1)[0]
        return self.URL.rstrip("/")

    def _validate_config(self) -> None:
        if not self.API_KEY:
            raise ValueError("VT_API_KEY is not set")
        if not self.URL:
            raise ValueError("VT_URL is not set")
    
if __name__ == "__main__":
    analyzer = UrlAnalyzer()
    report = analyzer.analyze_url_with_report("www.g00gle.com")
    print(report)
    
    
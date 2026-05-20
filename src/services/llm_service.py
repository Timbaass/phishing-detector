import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()


class LlmService:
    def __init__(self):
        self.api_url = os.environ["HF_API_URL"]
        self.hf_token = os.environ["HF_TOKEN"]
        self.model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct:together")

        self.headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }

    def query(self, payload: dict) -> dict:
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        return response.json()

    def generate_explanation(self, security_report: dict) -> dict:
        system_prompt = """
You are a cybersecurity explanation assistant for a phishing email detection system.

Your task is to explain the given structured security analysis to a non-technical user.

Rules:
- Do not change the final label.
- Do not invent new evidence.
- Use only the provided JSON data.
- If VirusTotal did not flag a URL, say "not flagged by VirusTotal", not "safe".
- If the ML model and VirusTotal disagree, explain the conflict clearly.
- Keep the explanation short, clear, and practical.
- Return JSON only.
"""

        user_prompt = f"""
Analyze this phishing detection result and generate a user-friendly explanation.

Security report JSON:
{json.dumps(security_report, indent=2)}

Return only valid JSON with this exact structure:

{{
  "title": "...",
  "risk_level": "Low | Medium | High",
  "summary": "...",
  "why_detected": [
    "...",
    "..."
  ],
  "url_analysis": [
    "..."
  ],
  "recommendation": "...",
  "technical_note": "..."
}}
"""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 700
        }

        result = self.query(payload)

        content = result["choices"][0]["message"]["content"]

        return self._safe_json_parse(content)

    def _safe_json_parse(self, content: str) -> dict:
        try:
            content = content.strip()

            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()

            return json.loads(content)

        except Exception:
            return {
                "title": "Email Security Analysis",
                "risk_level": "Unknown",
                "summary": "The system analyzed the email, but the AI explanation could not be generated properly.",
                "why_detected": [],
                "url_analysis": [],
                "recommendation": "Please review the technical detection results before taking action.",
                "technical_note": "LLM response could not be parsed as valid JSON."
            }
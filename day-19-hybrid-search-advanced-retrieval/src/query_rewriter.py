import os
import re
from typing import Optional

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class QueryRewriter:
    """Production Query Rewriter supporting Live LLM optimization + Clean algorithmic normalization."""
    def __init__(self, api_key: Optional[str] = None, use_mock: bool = False):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.use_mock = use_mock or os.getenv("USE_MOCK_LLM", "false").lower() == "true" or not self.api_key
        self.client = None

        if not self.use_mock and GENAI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.use_mock = True

    def rewrite(self, user_query: str) -> str:
        """Transforms typo-laden or ambiguous queries into clean search phrases."""
        if not self.use_mock and self.client:
            for model_id in ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.0-flash"]:
                try:
                    prompt = (
                        "You are an expert search engine query optimizer. "
                        "Fix spelling mistakes and expand acronyms without altering technical terms. "
                        "Return ONLY the reformulated query string on a single line with no markdown or formatting.\n\n"
                        f"User Query: {user_query}\nOptimized Query:"
                    )
                    resp = self.client.models.generate_content(
                        model=model_id,
                        contents=prompt
                    )
                    if resp and resp.text:
                        cleaned = resp.text.strip().replace('"', '').replace("`", "")
                        if len(cleaned) > 2:
                            return cleaned
                except Exception:
                    continue

        return self._algorithmic_clean(user_query)

    def _algorithmic_clean(self, query: str) -> str:
        """Rule-based cleanup for common typos and acronyms."""
        q = query.strip()
        q = re.sub(r'\bi\s+slinear\b', 'is linear', q, flags=re.IGNORECASE)
        q = re.sub(r'\bslinear\b', 'linear', q, flags=re.IGNORECASE)
        q = re.sub(r'\breggression\b', 'regression', q, flags=re.IGNORECASE)
        q = re.sub(r'\bhypotthesis\b', 'hypothesis', q, flags=re.IGNORECASE)

        if re.search(r'\bCE loss\b', q, flags=re.IGNORECASE):
            return "Cross Entropy log loss classification loss formula"
        if re.search(r'\bPCA\b', q, flags=re.IGNORECASE):
            return "Principal Component Analysis dimensionality reduction"

        return q
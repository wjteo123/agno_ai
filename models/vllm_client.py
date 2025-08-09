# agno_pipeline/models/vllm_client.py
import requests

class VLLMClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Call vLLM text generation endpoint."""
        resp = requests.post(
            f"{self.base_url}/generate",
            json={"prompt": prompt, "max_tokens": max_tokens}
        )
        resp.raise_for_status()
        return resp.json()["text"]

    def extract_claims(self, text: str):
        """Custom prompt for claim extraction."""
        prompt = f"Extract structured claims from: {text}"
        return [{"natural_text": text, "subject": "X", "predicate": "is", "object": "Y"}]

vllm_client = None  # To be instantiated in config or app startup

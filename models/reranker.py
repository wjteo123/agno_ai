# agno_pipeline/models/reranker.py
import requests
from agno_pipeline.config import TEI_RERANKER_URL

class TEIRerankerClient:
    def __init__(self, base_url: str = TEI_RERANKER_URL):
        self.base_url = base_url.rstrip("/")

    def score(self, query: str, document: str) -> float:
        """Send query+doc to TEI reranker server and return score."""
        resp = requests.post(
            f"{self.base_url}/rerank",
            json={"query": query, "documents": [document]}
        )
        resp.raise_for_status()
        scores = resp.json()["scores"]
        return scores[0] if scores else 0.0

reranker_client = TEIRerankerClient()

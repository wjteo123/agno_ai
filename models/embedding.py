# agno_pipeline/models/embedding.py
import requests
from agno_pipeline.config import TEI_EMBEDDING_URL

class TEIEmbeddingClient:
    def __init__(self, base_url: str = TEI_EMBEDDING_URL):
        self.base_url = base_url.rstrip("/")

    def embed(self, text: str) -> list:
        """Send a text to the TEI embedding server and return vector."""
        resp = requests.post(
            f"{self.base_url}/embed",
            json={"inputs": text}
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

embedding_client = TEIEmbeddingClient()

# agno_pipeline/db/qdrant_client.py
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from agno_pipeline.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION

class QdrantDBClient:
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if QDRANT_COLLECTION not in collections:
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )

    def upsert_fact(self, fact_id: str, vector: list, payload: dict):
        self.client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[PointStruct(id=fact_id, vector=vector, payload=payload)]
        )

    def query_vector(self, vector: list, top_k: int = 10):
        results = self.client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            limit=top_k
        )
        return results

    def delete_by_filter(self, filter_):
        self.client.delete(collection_name=QDRANT_COLLECTION, points_selector=filter_)

# Singleton instance
qdrant_client = QdrantDBClient()

# Helper for sync Celery tasks
def qdrant_sync_wrapper(func, *args, **kwargs):
    return func(*args, **kwargs)

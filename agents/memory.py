import asyncio
import logging
from agno_pipeline.db.mongo_client import mongo_client
from agno_pipeline.db.qdrant_client import qdrant_client
from agno_pipeline.models.embedding import embedding_client

logger = logging.getLogger("memory_agent")
logger.setLevel(logging.INFO)


async def admit_fact(fact_id: str) -> dict:
    """
    Memory Agent:
      - Ensure fact is in production state
      - Upsert into Qdrant (idempotent)
      - Keep provenance in Mongo
    """
    doc = await mongo_client.get_fact_by_id(fact_id)
    if not doc:
        logger.warning("Fact not found in memory agent: %s", fact_id)
        return {"fact_id": fact_id, "error": "not_found"}

    vec = await asyncio.to_thread(embedding_client.embed, doc["natural_text"])
    await asyncio.to_thread(qdrant_client.upsert_fact, fact_id, vec, doc)
    await mongo_client.insert_or_update_fact(fact_id, doc)
    return {"fact_id": fact_id, "status": doc.get("status")}

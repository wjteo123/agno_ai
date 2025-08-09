import asyncio
import logging
import time
from agno_pipeline.config import STAGING_CONFIRM_K, VERIFY_LOW_THRESHOLD
from agno_pipeline.db.mongo_client import mongo_client
from agno_pipeline.db.qdrant_client import qdrant_client
from agno_pipeline.models.embedding import embedding_client

logger = logging.getLogger("scoring_agent")
logger.setLevel(logging.INFO)


async def score_fact(fact_id: str) -> dict:
    """
    Scoring & Trust Agent:
      - Check for multiple confirmations (placeholder logic for now)
      - If confirmed, promote to production and boost trust
    """
    doc = await mongo_client.get_fact_by_id(fact_id)
    if not doc:
        logger.warning("Fact not found in scoring: %s", fact_id)
        return {"fact_id": fact_id, "error": "not_found"}

    confirmations = 2  # TODO: Implement real check across Qdrant or Mongo
    now = time.time()

    if confirmations >= STAGING_CONFIRM_K and doc["trust"] >= VERIFY_LOW_THRESHOLD:
        doc["status"] = "production"
        doc["trust"] = min(1.0, doc["trust"] + 0.05)
        vec = await asyncio.to_thread(embedding_client.embed, doc["natural_text"])
        await asyncio.to_thread(qdrant_client.upsert_fact, fact_id, vec, doc)
        doc["last_checked"] = now
        await mongo_client.insert_or_update_fact(fact_id, doc)
        return {"fact_id": fact_id, "admitted": True}

    doc["last_checked"] = now
    await mongo_client.insert_or_update_fact(fact_id, doc)
    return {"fact_id": fact_id, "admitted": False}

import time
import logging
from agno_pipeline.config import DECAY_HALF_LIFE_SECONDS, PRUNE_THRESHOLD
from agno_pipeline.db.mongo_client import mongo_client, mongo_sync_wrapper
from agno_pipeline.db.qdrant_client import qdrant_client

logger = logging.getLogger("pruning_agent")
logger.setLevel(logging.INFO)


def decay_trust(initial_trust: float, age_seconds: float) -> float:
    """Exponential decay formula."""
    half = DECAY_HALF_LIFE_SECONDS
    decay = 0.5 ** (age_seconds / half)
    return initial_trust * decay


async def prune_facts() -> dict:
    """
    Pruning Agent:
      - Decay trust based on last_checked/first_seen
      - Remove from DBs if trust < threshold
    """
    now = time.time()
    facts = await mongo_client.get_all_facts()
    pruned_ids = []

    for fact in facts:
        last_seen = fact.get("last_checked") or fact.get("first_seen") or now
        age = now - last_seen
        new_trust = decay_trust(fact.get("trust", 0), age)
        fact["trust"] = new_trust

        if new_trust < PRUNE_THRESHOLD:
            pruned_ids.append(fact["fact_id"])
            await mongo_client.delete_fact(fact["fact_id"])
            try:
                qdrant_client.delete_by_filter({"must": [{"key": "fact_id", "match": {"value": fact["fact_id"]}}]})
            except Exception:
                logger.exception("Failed to delete fact from Qdrant: %s", fact["fact_id"])
        else:
            await mongo_client.insert_or_update_fact(fact["fact_id"], fact)

    return {"pruned": len(pruned_ids), "ids": pruned_ids}

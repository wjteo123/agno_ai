# agno_pipeline/agents/ingestion.py
import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any

from agno_pipeline.config import QDRANT_COLLECTION
from agno_pipeline.db.mongo_client import mongo_client, mongo_sync_wrapper
from agno_pipeline.db.qdrant_client import qdrant_client, qdrant_sync_wrapper
from agno_pipeline.models.embedding import embedding_client
from agno_pipeline.models.vllm_client import vllm_client

logger = logging.getLogger("ingestion_agent")
logger.setLevel(logging.INFO)


def make_fact_id() -> str:
    return str(uuid.uuid4())


async def async_extract_claims(text: str) -> List[Dict[str, Any]]:
    """
    Call the (sync) vLLM client.extract_claims in a thread.
    This keeps agent async-friendly while reusing your existing vllm client.
    """
    return await asyncio.to_thread(vllm_client.extract_claims, text)


async def async_embed(text: str) -> List[float]:
    """
    Call the (sync) embedding TEI client in a thread.
    """
    return await asyncio.to_thread(embedding_client.embed, text)


async def async_upsert_qdrant(fact_id: str, vector: List[float], payload: Dict[str, Any]):
    return await asyncio.to_thread(qdrant_client.upsert_fact, fact_id, vector, payload)


async def async_insert_mongo(fact_id: str, doc: Dict[str, Any]):
    await mongo_client.insert_or_update_fact(fact_id, doc)


async def ingest_text(user_id: str, session_id: str, text: str, tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ingest text, extract claims, embed, persist to Qdrant+Mongo and return list of created facts.

    Returns: { 'created': [ { fact_id, natural_text }, ... ] }
    """
    ts = time.time()
    created = []
    try:
        claims = await async_extract_claims(text)
        logger.info("Extracted %d claims from input", len(claims))
        for c in claims:
            fact_id = make_fact_id()
            natural_text = c.get('natural_text', text)
            subject = c.get('subject')
            predicate = c.get('predicate')
            obj = c.get('object')

            # embedding (call sync TEI embedding in thread)
            try:
                vec = await async_embed(natural_text)
            except Exception as e:
                logger.exception("Embedding failed for fact %s: %s", fact_id, e)
                vec = [0.0] * 1536  # fallback (shouldn't happen in prod)

            doc = {
                'fact_id': fact_id,
                'natural_text': natural_text,
                'subject': subject,
                'predicate': predicate,
                'object': obj,
                'status': 'staging',
                'trust': 0.1,
                'sources': [{'type': 'chat', 'user_id': user_id, 'session_id': session_id, 'ts': ts}],
                'first_seen': ts,
                'last_checked': None,
            }

            # persist: qdrant (sync client in thread) & mongo (async)
            try:
                await async_upsert_qdrant(fact_id, vec, doc)
            except Exception:
                # try again as best-effort -- log but continue
                logger.exception("Qdrant upsert failed for %s", fact_id)

            await async_insert_mongo(fact_id, doc)
            created.append({'fact_id': fact_id, 'natural_text': natural_text})

    except Exception as e:
        logger.exception("ingest_text failed: %s", e)
        raise

    return {'created': created}

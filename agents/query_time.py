import asyncio
import logging
from typing import List, Dict
from agno_pipeline.db.qdrant_client import qdrant_client
from agno_pipeline.models.embedding import embedding_client
from agno_pipeline.models.reranker import reranker_client
from agno_pipeline.models.vllm_client import vllm_client

logger = logging.getLogger("query_time_agent")
logger.setLevel(logging.INFO)


async def retrieve_and_answer(user_query: str, top_k: int = 8) -> dict:
    """
    Query-Time Agent:
      - Embed the query
      - Retrieve top_k from Qdrant
      - Rerank
      - Build prompt for LLM
      - Generate answer
    """
    vec = await asyncio.to_thread(embedding_client.embed, user_query)
    hits = await asyncio.to_thread(qdrant_client.query_vector, vec, top_k)
    facts = [h.payload for h in hits if hasattr(h, "payload")]

    # Rerank facts with query
    reranked = []
    for f in facts:
        try:
            score = await asyncio.to_thread(reranker_client.score, user_query, f.get("natural_text", ""))
        except Exception:
            score = 0.0
        if score > 0:
            reranked.append((f, score))

    reranked.sort(key=lambda x: x[1], reverse=True)
    top_facts = [f for f, _ in reranked[:top_k]]

    # Build LLM prompt
    prompt = "Use the following facts to answer the query:\n"
    for f in top_facts:
        prompt += f"- {f.get('natural_text')} (trust={f.get('trust')})\n"
    prompt += f"\nUser Query: {user_query}\nAnswer:\n"

    answer = await asyncio.to_thread(vllm_client.generate, prompt)
    return {"answer": answer, "used_facts": [f.get("fact_id") for f in top_facts]}

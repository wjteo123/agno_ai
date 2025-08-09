# agno_pipeline/agents/verification.py
import asyncio
import logging
import time
from typing import List, Dict, Any
import httpx

from agno_pipeline.config import (
    SERPER_API_KEY,
    VERIFY_HIGH_THRESHOLD,
    VERIFY_LOW_THRESHOLD,
    RERANK_THRESH,
)
from agno_pipeline.db.mongo_client import mongo_client
from agno_pipeline.db.qdrant_client import qdrant_client
from agno_pipeline.models.reranker import reranker_client
from agno_pipeline.models.embedding import embedding_client
from agno_pipeline.models.vllm_client import vllm_client

logger = logging.getLogger("verification_agent")
logger.setLevel(logging.INFO)


async def fetch_serper_snippets(query: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Use Serper.dev search to obtain snippets.
    Returns list of {'snippet': str, 'link': str, 'title': str}
    """
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": top_n}
    url = "https://google.serper.dev/search"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    snippets = []
    # Serper's response shapes vary; try to extract organic results
    organic = data.get("organic", [])
    for item in organic[:top_n]:
        snippets.append({
            "snippet": item.get("snippet") or item.get("description") or "",
            "link": item.get("link"),
            "title": item.get("title")
        })
    return snippets


def compute_verification_score(max_rerank: float, consensus_frac: float, entail_prob: float, source_rel: float) -> float:
    w1, w2, w3, w4 = 0.4, 0.25, 0.25, 0.1
    return w1 * max_rerank + w2 * consensus_frac + w3 * entail_prob + w4 * source_rel


async def async_rerank_score(query: str, snippet: str) -> float:
    """
    Reranker client is sync; run in a thread.
    """
    return await asyncio.to_thread(reranker_client.score, query, snippet)


async def async_entailment_prob(claim: str, snippet: str) -> float:
    """
    Use vLLM to estimate entailment probability via a prompt.
    The vllm_client.generate is sync in current code, so execute in a thread.
    Returns a float in [0,1].
    """
    prompt = (
        "Rate how strongly the following snippet supports the claim on a scale 0.0-1.0.\n\n"
        f"Claim: {claim}\n\nSnippet: {snippet}\n\n"
        "Answer with a single number between 0.0 and 1.0 (e.g., 0.75)."
    )
    try:
        resp_text = await asyncio.to_thread(vllm_client.generate, prompt, 32)
        # try to extract a float
        for token in resp_text.split():
            try:
                v = float(token.strip().strip('.,'))  # simple parse
                # clamp
                return max(0.0, min(1.0, v))
            except Exception:
                continue
    except Exception:
        logger.exception("Entailment call failed")
    # fallback
    return 0.5


async def verify_fact(fact_id: str) -> Dict[str, Any]:
    """
    Verification flow:
      - fetch fact from Mongo
      - search web via Serper
      - rerank snippets and compute entailment
      - compute verification score and update Mongo + Qdrant
    """
    now = time.time()
    doc = await mongo_client.get_fact_by_id(fact_id)
    if not doc:
        logger.warning("verify_fact: fact not found %s", fact_id)
        return {"fact_id": fact_id, "error": "not_found"}

    claim_text = doc.get("natural_text")

    # fetch web snippets
    try:
        snippets = await fetch_serper_snippets(claim_text, top_n=6)
    except Exception:
        logger.exception("Serper search failed for fact %s", fact_id)
        snippets = []

    if not snippets:
        # minimal update: set last_checked and leave staging
        doc['last_checked'] = now
        await mongo_client.insert_or_update_fact(fact_id, doc)
        return {"fact_id": fact_id, "score": 0.0, "status": doc.get('status')}

    rerank_scores = []
    entail_probs = []
    domains = set()
    # iterate snippets
    for s in snippets:
        snippet_text = s.get("snippet", "")
        # rerank (sync TEI client) in thread
        try:
            rscore = await async_rerank_score(claim_text, snippet_text)
        except Exception:
            logger.exception("Reranker failed for %s", fact_id)
            rscore = 0.0
        rerank_scores.append(rscore)
        # entailment via vLLM
        try:
            ep = await async_entailment_prob(claim_text, snippet_text)
        except Exception:
            logger.exception("Entailment failed for %s", fact_id)
            ep = 0.0
        entail_probs.append(ep)
        link = s.get("link")
        if link:
            domains.add(link.split("/")[2] if "/" in link else link)

    max_rerank = max(rerank_scores) if rerank_scores else 0.0
    consensus_frac = len(domains) / max(1, len(snippets))
    entail_prob = max(entail_probs) if entail_probs else 0.0
    source_reliability = 0.6  # TODO: compute via domain reputation service

    score = compute_verification_score(max_rerank, consensus_frac, entail_prob, source_reliability)
    # update doc
    doc['last_checked'] = now
    doc['last_verification_score'] = float(score)
    doc.setdefault('sources', []).append({'type': 'web_check', 'score': float(score), 'ts': now, 'n_snippets': len(snippets)})

    # thresholding
    if score >= VERIFY_HIGH_THRESHOLD:
        doc['status'] = 'production'
        doc['trust'] = float(score)
        # recompute embedding and upsert to qdrant to ensure production vector is fresh
        try:
            vec = await asyncio.to_thread(embedding_client.embed, claim_text)
            await asyncio.to_thread(qdrant_client.upsert_fact, fact_id, vec, doc)
        except Exception:
            logger.exception("Failed to upsert to qdrant during verification for %s", fact_id)
    elif score > VERIFY_LOW_THRESHOLD:
        doc['status'] = 'staging'
        doc['trust'] = float(score)
    else:
        doc['status'] = 'rejected'
        doc['trust'] = float(score)

    # persist to mongo
    await mongo_client.insert_or_update_fact(fact_id, doc)

    return {"fact_id": fact_id, "score": float(score), "status": doc['status']}

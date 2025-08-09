# agno_pipeline/main.py
import time
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from agno_pipeline.tasks.pipeline_tasks import (
    ingest_claims_task, verify_claim_task, score_claim_task,
    admit_claim_task, prune_facts_task
)
from agno_pipeline.agents.query_time import retrieve_and_answer
from agno_pipeline.db.mongo_client import mongo_client
from agno_pipeline.db.qdrant_client import qdrant_client

logger = logging.getLogger("agno_main")
logger.setLevel(logging.INFO)

app = FastAPI(title="Agno Multi-Agent Autonomous Pipeline")

# ----------- API Schemas -----------
class IngestPayload(BaseModel):
    user_id: str
    session_id: str
    text: str
    tools: List[Dict[str, Any]] = None
    timestamp: float = None

class QueryPayload(BaseModel):
    user_id: str
    session_id: str
    query: str
    top_k: int = 8

class FactIDPayload(BaseModel):
    fact_id: str

# ----------- Endpoints -----------
@app.post("/ingest")
def api_ingest(payload: IngestPayload):
    payload.timestamp = payload.timestamp or time.time()
    task = ingest_claims_task.delay(payload.dict())
    return {"status": "accepted", "task_id": task.id}

@app.post("/verify")
def api_verify(payload: FactIDPayload):
    task = verify_claim_task.delay(payload.dict())
    return {"status": "accepted", "task_id": task.id}

@app.post("/score")
def api_score(payload: FactIDPayload):
    task = score_claim_task.delay(payload.dict())
    return {"status": "accepted", "task_id": task.id}

@app.post("/admit")
def api_admit(payload: FactIDPayload):
    task = admit_claim_task.delay(payload.dict())
    return {"status": "accepted", "task_id": task.id}

@app.post("/prune")
def api_prune():
    task = prune_facts_task.delay({})
    return {"status": "prune_scheduled", "task_id": task.id}

@app.post("/query")
async def api_query(payload: QueryPayload):
    result = await retrieve_and_answer(payload.query, payload.top_k)
    return result

@app.get("/admin/stats")
async def api_stats():
    facts_count = len(await mongo_client.get_all_facts())
    return {
        "mongo_facts": facts_count,
        "qdrant_points": len(qdrant_client.query_vector([0]*1536, 1))  # just a ping
    }

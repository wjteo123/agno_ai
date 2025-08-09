# agno_pipeline/tasks/pipeline_tasks.py
import asyncio
from agno_pipeline.tasks.celery_app import celery_app
from agno_pipeline.agents.ingestion import ingest_text
from agno_pipeline.agents.verification import verify_fact
from agno_pipeline.agents.scoring import score_fact
from agno_pipeline.agents.memory import admit_fact
from agno_pipeline.agents.pruning import prune_facts

@celery_app.task(name='tasks.ingest_claims')
def ingest_claims_task(payload: dict):
    return asyncio.run(
        ingest_text(payload['user_id'], payload['session_id'], payload['text'], payload.get('tools'))
    )

@celery_app.task(name='tasks.verify_claim')
def verify_claim_task(payload: dict):
    return asyncio.run(verify_fact(payload['fact_id']))

@celery_app.task(name='tasks.score_claim')
def score_claim_task(payload: dict):
    return asyncio.run(score_fact(payload['fact_id']))

@celery_app.task(name='tasks.admit_claim')
def admit_claim_task(payload: dict):
    return asyncio.run(admit_fact(payload['fact_id']))

@celery_app.task(name='tasks.prune_facts')
def prune_facts_task(payload: dict = None):
    return asyncio.run(prune_facts())

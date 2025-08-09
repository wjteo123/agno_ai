# agno_pipeline/tasks/celery_app.py
import os
from celery import Celery

REDIS_BROKER = os.getenv('REDIS_BROKER', 'redis://localhost:6379/0')
CELERY_BACKEND = os.getenv('CELERY_BACKEND', 'redis://localhost:6379/1')

celery_app = Celery(
    'agno_pipeline',
    broker=REDIS_BROKER,
    backend=CELERY_BACKEND
)

celery_app.conf.task_routes = {
    'tasks.ingest_claims': {'queue': 'ingest'},
    'tasks.verify_claim': {'queue': 'verify'},
    'tasks.score_claim': {'queue': 'score'},
    'tasks.admit_claim': {'queue': 'admit'},
    'tasks.prune_facts': {'queue': 'prune'},
}
celery_app.conf.worker_prefetch_multiplier = 1

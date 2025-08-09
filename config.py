# agno_pipeline/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "agno")

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "facts")

# Serper API
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# TEI Servers
TEI_EMBEDDING_URL = os.getenv("TEI_EMBEDDING_URL", "http://tei-embedding:8080")
TEI_RERANKER_URL = os.getenv("TEI_RERANKER_URL", "http://tei-reranker:8080")

# Verification thresholds
VERIFY_HIGH_THRESHOLD = float(os.getenv("VERIFY_HIGH_THRESHOLD", 0.85))
VERIFY_LOW_THRESHOLD = float(os.getenv("VERIFY_LOW_THRESHOLD", 0.55))
RERANK_THRESH = float(os.getenv("RERANK_THRESH", 0.7))
STAGING_CONFIRM_K = int(os.getenv("STAGING_CONFIRM_K", 2))
STAGING_WINDOW_SECONDS = int(os.getenv("STAGING_WINDOW_SECONDS", 48 * 3600))
DECAY_HALF_LIFE_SECONDS = int(os.getenv("DECAY_HALF_LIFE_SECONDS", 30 * 24 * 3600))
PRUNE_THRESHOLD = float(os.getenv("PRUNE_THRESHOLD", 0.15))

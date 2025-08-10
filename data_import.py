from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from pymongo import MongoClient
from datetime import datetime

# ========================
# EMBEDDING & RERANKER TEMPLATES
# ========================

EMBEDDING_INSTRUCTION = (
    "Represent the legal query or clause for semantic search. "
    "Focus on legal meaning and jurisdiction-specific context. "
    "Return a single vector representation."
)

def format_for_embedding(query: str) -> str:
    return f"{EMBEDDING_INSTRUCTION} {query} <|endoftext|>"

RERANKER_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. The answer can only be 'yes' or 'no'."
)

def format_for_reranker(instruction: str, query: str, document: str) -> str:
    return (
        f"<|im_start|>system\n{RERANKER_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

# ========================
# MONGODB SETUP
# ========================

def setup_mongodb():
    print("📦 Setting up MongoDB collections...")
    mongo_client = MongoClient("mongodb://localhost:27017/")
    db = mongo_client["legal_ai_db"]

    collections = [
        "audit_logs",
        "prompts",
        "tools",
        "learning_tasks",
        "agents_stats",
        "source_registry"
    ]

    for name in collections:
        if name not in db.list_collection_names():
            db.create_collection(name)
            print(f"  ✅ Created collection: {name}")
        else:
            print(f"  ℹ️ Collection {name} already exists")

    # Indexes
    db.audit_logs.create_index([("timestamp", 1)])
    db.prompts.create_index([("name", 1)], unique=True)
    db.tools.create_index([("name", 1)], unique=True)
    db.learning_tasks.create_index([("status", 1)])
    db.agents_stats.create_index([("agent_name", 1), ("date", 1)])
    db.source_registry.create_index([("source_id", 1)], unique=True)

    # Sample log
    db.audit_logs.insert_one({
        "timestamp": datetime.utcnow().isoformat(),
        "agent": "system_init",
        "action": "db_setup",
        "details": "Initial collections created"
    })
    print("📦 MongoDB setup complete.\n")
    return db

# ========================
# QDRANT SETUP
# ========================

def setup_qdrant():
    print("📦 Setting up Qdrant collections...")
    client = QdrantClient(host="localhost", port=6333)
    vector_params = VectorParams(size=4096, distance=Distance.COSINE)

    collections = [
        "legal_docs",
        "prompt_vectors",
        "tool_vectors"
    ]

    for name in collections:
        if not client.collection_exists(name):
            client.recreate_collection(
                collection_name=name,
                vectors_config=vector_params
            )
            print(f"  ✅ Created collection: {name} with 4096-dim vectors")
        else:
            print(f"  ℹ️ Collection {name} already exists")

    print("📦 Qdrant setup complete.\n")
    return client

# ========================
# VERIFICATION
# ========================

def verify_datastores(db, qdrant_client):
    print("🔍 Verifying MongoDB collections...")
    for name in db.list_collection_names():
        count = db[name].count_documents({})
        print(f"  📂 {name}: {count} documents")

    print("\n🔍 Verifying Qdrant collections...")
    for coll in ["legal_docs", "prompt_vectors", "tool_vectors"]:
        info = qdrant_client.get_collection(coll)
        print(f"  📂 {coll}: {info.points_count} vectors, dim=4096")

# ========================
# MAIN EXECUTION
# ========================

if __name__ == "__main__":
    db = setup_mongodb()
    qdrant_client = setup_qdrant()
    verify_datastores(db, qdrant_client)

    print("\n📝 Embedding template example:")
    print(format_for_embedding("What is the penalty for breach of contract in Malaysia?"))

    print("\n📝 Reranker template example:")
    print(format_for_reranker(
        "Retrieve relevant Malaysian contract law clauses",
        "Penalty for breach of contract",
        "Section 74 of the Contracts Act 1950 states..."
    ))

    print("\n✅ All datastores initialized, verified, and templates ready.")

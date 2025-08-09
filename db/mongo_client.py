# agno_pipeline/db/mongo_client.py
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from agno_pipeline.config import MONGO_URI, MONGO_DB

class MongoDBClient:
    def __init__(self):
        self.client = AsyncIOMotorClient(MONGO_URI)
        self.db = self.client[MONGO_DB]
        self.facts = self.db["facts"]

    async def insert_or_update_fact(self, fact_id: str, doc: dict):
        await self.facts.update_one(
            {"fact_id": fact_id},
            {"$set": doc},
            upsert=True
        )

    async def get_fact_by_id(self, fact_id: str):
        return await self.facts.find_one({"fact_id": fact_id})

    async def find_facts_by_subject(self, subject: str):
        cursor = self.facts.find({"subject": subject})
        return await cursor.to_list(length=None)

    async def delete_fact(self, fact_id: str):
        await self.facts.delete_one({"fact_id": fact_id})

    async def get_all_facts(self):
        cursor = self.facts.find({})
        return await cursor.to_list(length=None)

# Singleton instance
mongo_client = MongoDBClient()

# Helper for sync Celery tasks
def mongo_sync_wrapper(coro):
    return asyncio.run(coro)

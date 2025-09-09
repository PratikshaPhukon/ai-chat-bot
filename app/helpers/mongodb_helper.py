import os
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Dict
from fastapi import HTTPException
from app.models.resource_model import Resource
from app.models.user_model import User
import asyncio


class MultiTenantMongoDBHelper:
    """Multi-tenant MongoDB helper with connection pooling."""

    def __init__(self):
        self.clients: Dict[str, AsyncIOMotorClient] = {}
        self.initialized_databases: Dict[str, bool] = {}
        self._lock = asyncio.Lock()
        self._connection_pool_size = int(os.getenv("MONGO_CONNECTION_POOL_SIZE", "10"))
        self._max_pool_size = int(os.getenv("MONGO_MAX_POOL_SIZE", "50"))

    async def _get_mongo_url(self) -> str:
        """Get MongoDB connection URL from environment variables."""
        username = os.getenv("MONGO_DB_USERNAME")
        password = os.getenv("MONGO_DB_PASSWORD")
        host = os.getenv("MONGO_DB_HOST")

        if not all([username, password, host]):
            raise ValueError("Missing MongoDB environment variables")

        return f"mongodb+srv://{username}:{password}@{host}/?authSource=admin"

    async def _get_client_for_tenant(self, org_id: str) -> AsyncIOMotorClient:
        """Get or create a MongoDB client for a specific tenant."""

        # Fast path if already exists
        if org_id in self.clients:
            return self.clients[org_id]

        mongo_url = await self._get_mongo_url()

        # Create client with connection pooling
        client = AsyncIOMotorClient(
            mongo_url,
            maxPoolSize=self._max_pool_size,
            minPoolSize=self._connection_pool_size,
            maxIdleTimeMS=30000,  # 30 seconds
            waitQueueTimeoutMS=5000,  # 5 seconds
            retryWrites=True,
            retryReads=True,
        )

        try:
            await client.admin.command("ping")

            async with self._lock:
                if org_id not in self.clients:
                    self.clients[org_id] = client
                    print(f"Created MongoDB client for tenant: {org_id}")
        except Exception as e:
            client.close()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to MongoDB for tenant {org_id}: {str(e)}",
            )

        return self.clients[org_id]

    async def _initialize_database_for_tenant(self, org_id: str):
        """Initialize Beanie for a specific tenant database."""
        if self.initialized_databases.get(org_id):
            return

        client = await self._get_client_for_tenant(org_id)

        async with self._lock:
            if not self.initialized_databases.get(org_id):
                await init_beanie(
                    database=client[org_id], document_models=[User, Resource]
                )
                self.initialized_databases[org_id] = True
                print(f"Initialized database for tenant: {org_id}")

    async def _ensure_tenant_initialized(self, org_id: str):
        """Ensure database is initialized for a specific tenant."""
        if not org_id:
            raise HTTPException(status_code=400, detail="Organization ID is required")

        await self._initialize_database_for_tenant(org_id)

    async def close_all_connections(self):
        """Close all database connections."""
        async with self._lock:
            for org_id, client in self.clients.items():
                try:
                    client.close()
                    print(f"Closed MongoDB connection for tenant: {org_id}")
                except Exception as e:
                    print(f"Error closing connection for tenant {org_id}: {e}")
            self.clients.clear()
            self.initialized_databases.clear()
            print("All MongoDB connections closed.")


# Global instance
mongodb_helper = MultiTenantMongoDBHelper()


# --- Optional compatibility layer --- #
async def connect_to_mongo():
    """Initialize MongoDB connection (for backward compatibility)."""
    print(
        "Multi-tenant MongoDB helper initialized. Connections will be created per tenant."
    )


async def close_mongo_connection():
    """Close all database connections."""
    await mongodb_helper.close_all_connections()
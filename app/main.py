import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from app.routes.user_route import router as user_router
from app.routes.resource_route import router as resource_router
from app.routes.chatbot_route import router as chatbot_router
from app.middleware.logging_middleware import LoggingMiddleware
from app.middleware.auth_middleware import JWTAuthMiddleware
from app.helpers.mongodb_helper import connect_to_mongo, close_mongo_connection

# Load environment configuration
load_dotenv()

# Environment-based configuration
ENV = os.getenv("ENV", "development")
if ENV == "production":
    load_dotenv(".env.production")
    allowed_origins = [
        "https://learn.calibr.ai",
    ]
else:
    load_dotenv(".env")
    allowed_origins = ["http://localhost:4201", "https://learn.calibr.ai"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management for startup and shutdown events."""
    # Startup
    print("Starting up Multi-Agent Chatbot API...")
    await connect_to_mongo()
    print("MongoDB connection initialized")
    
    yield
    
    # Shutdown
    print("Shutting down Multi-Agent Chatbot API...")
    await close_mongo_connection()
    print("All connections closed")


# Initialize FastAPI application
app = FastAPI(
    title="Multi-Agent Chatbot API",
    description="Production-ready chatbot with multi-agent architecture for educational content retrieval",
    version="1.0.0",
    docs_url="/docs" if ENV == "development" else None,
    redoc_url="/redoc" if ENV == "development" else None,
    lifespan=lifespan
)

# Register middleware (order matters!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)
app.add_middleware(JWTAuthMiddleware)

# Register API routers
app.include_router(
    user_router, 
    prefix="/api/v1/users", 
    tags=["User Management"]
)
app.include_router(
    resource_router, 
    prefix="/api/v1/resources", 
    tags=["Resource Management"]
)
app.include_router(
    chatbot_router, 
    prefix="/api/v1/chatbot", 
    tags=["Chatbot Interaction"]
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Multi-Agent Chatbot API is running",
        "version": "1.0.0",
        "environment": ENV
    }


@app.get("/health")
async def health_check():
    """Detailed health check for monitoring."""
    return {
        "status": "healthy",
        "environment": ENV,
        "services": {
            "mongodb": "connected",
            "qdrant": "configured",
            "vertex_ai": "configured"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=ENV == "development"
    )
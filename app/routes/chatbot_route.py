"""
Chatbot Routes - FastAPI endpoints for multi-agent chatbot interactions
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.controllers.chatbot_controller import ask_question_stream_controller, get_pipeline_health

router = APIRouter()


class ChatbotRequest(BaseModel):
    """Request model for chatbot interactions."""
    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    course_id: Optional[str] = Field(None, description="Optional course context ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What courses are available in machine learning?",
                "course_id": "course_123"
            }
        }


class ChatbotResponse(BaseModel):
    """Response model for chatbot interactions (for documentation)."""
    code: int = Field(description="Response status code")
    status: str = Field(description="Response status")
    message: str = Field(description="Response message")
    data: Optional[dict] = Field(None, description="Response data")


@router.post(
    "/ask/stream",
    summary="Ask chatbot question with streaming response",
    description="Submit a question to the multi-agent chatbot system and receive a streaming response",
    response_description="Streaming text response from the chatbot",
    tags=["Chatbot"]
)
async def ask_question_stream(
    request: Request, 
    chatbot_request: ChatbotRequest
):
    """
    Ask a question to the multi-agent chatbot system.
    
    This endpoint processes queries through a sophisticated multi-agent pipeline:
    1. Query refinement and normalization
    2. Intent analysis and routing
    3. Database query execution (if needed)
    4. Semantic content retrieval (if needed)  
    5. Post-processing and optimization
    6. LLM-powered response synthesis
    
    The response is streamed in real-time for better user experience.
    """
    try:
        return await ask_question_stream_controller(
            request=request,
            question=chatbot_request.question,
            course_id=chatbot_request.course_id
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch any unexpected exceptions
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error in chatbot interaction: {str(e)}"
        )


@router.get(
    "/health",
    summary="Check chatbot system health",
    description="Get health status of the multi-agent chatbot pipeline",
    response_model=dict,
    tags=["System Health"]
)
async def chatbot_health_check():
    """
    Check the health status of the chatbot system.
    
    Returns information about:
    - Overall system health
    - Individual agent status
    - Configuration parameters
    - Performance metrics
    """
    return await get_pipeline_health()


@router.get(
    "/agents/status",
    summary="Get detailed agent status",
    description="Get detailed status information about all agents in the pipeline",
    tags=["System Health"]
)
async def get_agents_status():
    """Get detailed status of all agents in the multi-agent pipeline."""
    from app.controllers.chatbot_controller import orchestrator
    
    return {
        "pipeline_status": orchestrator.get_pipeline_status(),
        "agents": {
            "query_refinement": {
                "description": "Normalizes queries and injects chat history",
                "status": "active"
            },
            "intent_router": {
                "description": "Routes queries to appropriate processing paths",
                "status": "active"
            },
            "mongodb_agent": {
                "description": "Handles structured database queries",
                "status": "active"
            },
            "filter_agent": {
                "description": "Extracts metadata filters for vector search",
                "status": "active"
            },
            "rag_agent": {
                "description": "Performs semantic content retrieval",
                "status": "active"
            },
            "post_retrieval_processor": {
                "description": "Processes and optimizes retrieved content",
                "status": "active"
            },
            "llm_synthesis_agent": {
                "description": "Synthesizes final responses",
                "status": "active"
            }
        }
    }
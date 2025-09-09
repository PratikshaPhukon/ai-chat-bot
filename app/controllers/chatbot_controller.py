"""
Multi-Agent Chatbot Controller
Orchestrates the complete multi-agent pipeline for chatbot interactions.
"""

from fastapi import Request, HTTPException
from typing import Optional
import asyncio
import logging
from fastapi.responses import StreamingResponse
import json

# Import all agents
from app.services.agents.query_refinement_agent import QueryRefinementAgent
from app.services.agents.intent_router_agent import IntentRouterAgent
from app.services.agents.mongodb_agent import MongoDBAgent
from app.services.agents.filter_agent import FilterAgent
from app.services.agents.rag_agent import RAGAgent
from app.services.agents.post_retrieval_processor import PostRetrievalProcessor
from app.services.agents.llm_synthesis_agent import LLMSynthesisAgent

# Import helpers and middleware
from app.helpers.vertex_helper import get_credentials, initialize_vertex_ai
from app.middleware.auth_middleware import get_user_info_from_request

logger = logging.getLogger(__name__)


class ChatbotOrchestrator:
    """
    Main orchestrator for the multi-agent chatbot system.
    Coordinates the flow between all agents in the pipeline.
    """
    
    def __init__(self):
        # Initialize all agents
        self.query_refinement = QueryRefinementAgent()
        self.intent_router = IntentRouterAgent()
        self.mongodb_agent = MongoDBAgent()
        self.filter_agent = FilterAgent()
        self.rag_agent = RAGAgent()
        self.post_processor = PostRetrievalProcessor()
        self.llm_synthesis = LLMSynthesisAgent()
        
        # Pipeline configuration
        self.max_pipeline_timeout = 60  # seconds
        self.enable_parallel_processing = True
    
    async def process_query(
        self,
        user_query: str,
        user_context: dict,
        chat_history: Optional[list] = None,
        course_id: Optional[str] = None
    ) -> StreamingResponse:
        """
        Main entry point for query processing through the multi-agent pipeline.
        
        Args:
            user_query: Raw user input
            user_context: User authentication and context info
            chat_history: Previous conversation history
            course_id: Optional course context
            
        Returns:
            StreamingResponse for real-time interaction
        """
        try:
            # Step 1: Query Refinement
            logger.info(f"Starting multi-agent pipeline for query: {user_query[:100]}")
            
            refined_query = self.query_refinement.refine_query(
                user_query=user_query,
                chat_history=chat_history,
                user_context=user_context
            )
            
            if "error" in refined_query:
                return self._create_error_stream(refined_query["error"])
            
            # Step 2: Intent Routing
            routing_decision = self.intent_router.route_query(refined_query)
            
            if "error" in routing_decision.get("routing_metadata", {}):
                return self._create_error_stream("Query routing failed")
            
            # Step 3: Execute Processing Pipeline
            processing_path = routing_decision["processing_path"]
            instructions = routing_decision["processing_instructions"]
            
            # Add course context if provided
            if course_id:
                instructions["course_context"] = course_id
                user_context["course_id"] = course_id
            
            logger.info(f"Processing path determined: {processing_path}")
            
            # Execute based on routing decision
            if processing_path == "conversational":
                processed_content = await self._handle_conversational_path(instructions, user_context)
            elif processing_path == "operational":
                processed_content = await self._handle_operational_path(instructions, user_context)
            elif processing_path == "semantic":
                processed_content = await self._handle_semantic_path(instructions, user_context)
            elif processing_path == "hybrid":
                processed_content = await self._handle_hybrid_path(instructions, user_context)
            else:
                processed_content = {"success": False, "error": "Unknown processing path"}
            
            # Step 4: Generate Streaming Response
            return StreamingResponse(
                self._generate_streaming_response(processed_content, instructions, user_context),
                media_type="text/plain"
            )
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return self._create_error_stream(f"System error: {str(e)}")
    
    async def _handle_conversational_path(self, instructions: dict, user_context: dict) -> dict:
        """Handle conversational queries (greetings, thank you, etc.)."""
        logger.info("Processing conversational query")
        
        return {
            "success": True,
            "processed_content": {
                "response_type": "conversational",
                "context": "This is a conversational interaction.",
                "direct_response": True
            },
            "content_type": "conversational",
            "processing_metadata": {
                "agents_used": ["query_refinement", "intent_router", "llm_synthesis"],
                "processing_time": "minimal"
            }
        }
    
    async def _handle_operational_path(self, instructions: dict, user_context: dict) -> dict:
        """Handle operational queries requiring database access."""
        logger.info("Processing operational query - accessing database")
        
        # Execute MongoDB query
        mongodb_results = await self.mongodb_agent.execute_query(instructions, user_context)
        
        # Process results for LLM consumption
        processed_content = self.post_processor.process_retrieved_content(
            rag_results={},  # No RAG results for pure operational queries
            mongodb_results=mongodb_results,
            instructions=instructions
        )
        
        return processed_content
    
    async def _handle_semantic_path(self, instructions: dict, user_context: dict) -> dict:
        """Handle semantic queries requiring content retrieval."""
        logger.info("Processing semantic query - vector search pipeline")
        
        # Step 1: Extract filters
        filter_results = self.filter_agent.extract_filters(instructions)
        
        # Step 2: Perform RAG retrieval
        rag_results = await self.rag_agent.retrieve_content(
            instructions=instructions,
            filter_payload=filter_results,
            user_context=user_context
        )
        
        # Step 3: Process and optimize content
        processed_content = self.post_processor.process_retrieved_content(
            rag_results=rag_results,
            mongodb_results=None,
            instructions=instructions
        )
        
        return processed_content
    
    async def _handle_hybrid_path(self, instructions: dict, user_context: dict) -> dict:
        """Handle hybrid queries requiring both database and content retrieval."""
        logger.info("Processing hybrid query - parallel database and vector search")
        
        if self.enable_parallel_processing:
            # Execute MongoDB and RAG operations in parallel
            mongodb_task = asyncio.create_task(
                self.mongodb_agent.execute_query(instructions, user_context)
            )
            
            # Extract filters first
            filter_results = self.filter_agent.extract_filters(instructions)
            
            rag_task = asyncio.create_task(
                self.rag_agent.retrieve_content(
                    instructions=instructions,
                    filter_payload=filter_results,
                    user_context=user_context
                )
            )
            
            # Wait for both operations to complete
            try:
                mongodb_results, rag_results = await asyncio.gather(
                    mongodb_task, rag_task, timeout=self.max_pipeline_timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Hybrid query processing timed out")
                return {"success": False, "error": "Query processing timed out"}
            
        else:
            # Sequential processing
            mongodb_results = await self.mongodb_agent.execute_query(instructions, user_context)
            
            filter_results = self.filter_agent.extract_filters(instructions)
            
            rag_results = await self.rag_agent.retrieve_content(
                instructions=instructions,
                filter_payload=filter_results,
                user_context=user_context
            )
        
        # Process and merge results
        processed_content = self.post_processor.process_retrieved_content(
            rag_results=rag_results,
            mongodb_results=mongodb_results,
            instructions=instructions
        )
        
        return processed_content
    
    async def _generate_streaming_response(
        self, 
        processed_content: dict, 
        instructions: dict, 
        user_context: dict
    ):
        """Generate streaming response using LLM synthesis."""
        
        try:
            # Get the response generator from LLM synthesis agent
            response_generator = self.llm_synthesis.synthesize_response(
                processed_content=processed_content,
                instructions=instructions,
                user_context=user_context
            )
            
            # Check if it's an async generator or regular generator
            if hasattr(response_generator, '__aiter__'):
                # It's an async generator
                async for chunk in response_generator:
                    yield chunk
            else:
                # It's a regular generator, iterate normally but yield in async context
                for chunk in response_generator:
                    yield chunk
                    # Add a small delay to allow other coroutines to run
                    await asyncio.sleep(0)
                    
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            yield f"\n\nI apologize, but I encountered an error generating the response: {str(e)}"
    
    def _create_error_stream(self, error_message: str) -> StreamingResponse:
        """Create error response in streaming format."""
        
        async def error_generator():
            error_response = f"I apologize, but I encountered an issue: {error_message}"
            words = error_response.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.05)  # Small delay for streaming effect
        
        return StreamingResponse(error_generator(), media_type="text/plain")
    
    def get_pipeline_status(self) -> dict:
        """Get current status of the pipeline for monitoring."""
        return {
            "agents_status": {
                "query_refinement": "active",
                "intent_router": "active", 
                "mongodb_agent": "active",
                "filter_agent": "active",
                "rag_agent": "active",
                "post_processor": "active",
                "llm_synthesis": "active"
            },
            "configuration": {
                "max_timeout": self.max_pipeline_timeout,
                "parallel_processing": self.enable_parallel_processing
            }
        }


# Global orchestrator instance
orchestrator = ChatbotOrchestrator()


async def ask_question_stream_controller(
    request: Request,
    question: str,
    course_id: Optional[str] = None
):
    """
    Main controller function for chatbot interactions.
    
    Args:
        request: FastAPI request object
        question: User's question
        course_id: Optional course context
        
    Returns:
        StreamingResponse with chatbot answer
    """
    try:
        # Extract user context from request
        user_id, org_id = get_user_info_from_request(request)
        
        user_context = {
            "user_id": user_id,
            "org_id": org_id,
            "course_id": course_id
        }
        
        logger.info(f"Processing chatbot question for org_id={org_id}, user_id={user_id}")
        
        # Initialize Vertex AI credentials
        credentials, token_credentials = get_credentials()
        initialize_vertex_ai(credentials)
        
        # Add credentials to user context
        user_context["credentials"] = credentials
        user_context["token_credentials"] = token_credentials
        
        # Process through multi-agent pipeline
        response = await orchestrator.process_query(
            user_query=question,
            user_context=user_context,
            chat_history=None,  # Could be enhanced to maintain chat history
            course_id=course_id
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chatbot controller: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


async def get_pipeline_health():
    """Health check endpoint for the multi-agent pipeline."""
    try:
        status = orchestrator.get_pipeline_status()
        return {
            "status": "healthy",
            "pipeline": status,
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": asyncio.get_event_loop().time()
        }
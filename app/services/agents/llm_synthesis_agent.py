"""
LLM Synthesis Agent - Final Step of Multi-Agent Pipeline
Combines processed context and generates coherent, final responses.
"""

from typing import Dict, List, Optional, Any, Generator
import logging
from app.helpers.vertex_helper import generate_answer_stream

logger = logging.getLogger(__name__)


class LLMSynthesisAgent:
    """
    Final agent responsible for synthesizing processed content into coherent responses:
    - Combines retrieved context with original query
    - Generates contextually appropriate responses
    - Formats output for streaming delivery
    - Handles different response types (conversational, informational, hybrid)
    """
    
    def __init__(self):
        self.max_response_length = 2000
        self.streaming_enabled = True
        self.response_templates = self._initialize_response_templates()
        
    def synthesize_response(
        self,
        processed_content: Dict,
        instructions: Dict,
        user_context: Dict
    ) -> Generator[str, None, None]:
        """
        Main synthesis function that generates streaming responses.
        
        Args:
            processed_content: Output from PostRetrievalProcessor
            instructions: Original processing instructions
            user_context: User context information
            
        Yields:
            Response chunks for streaming
        """
        try:
            # Validate inputs
            if not processed_content.get("success", False):
                error_message = processed_content.get("error", "Content processing failed")
                yield from self._generate_error_response(error_message)
                return
            
            # Extract synthesis parameters
            response_type = processed_content.get("content_type", "semantic")
            original_query = instructions.get("original_query", instructions.get("query", ""))
            
            # Build context for LLM
            context = self._build_llm_context(processed_content, instructions, user_context)
            
            # Create appropriate prompt
            prompt = self._create_synthesis_prompt(
                context, original_query, response_type, instructions
            )
            
            # Generate streaming response
            if self.streaming_enabled:
                yield from generate_answer_stream(context, original_query)
            else:
                # Non-streaming fallback (not implemented in original helpers)
                yield self._generate_complete_response(context, original_query)
                
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            yield from self._generate_error_response(f"Response generation failed: {str(e)}")
    
    def _build_llm_context(
        self,
        processed_content: Dict,
        instructions: Dict,
        user_context: Dict
    ) -> str:
        """Build comprehensive context string for LLM input."""
        
        context_parts = []
        content_data = processed_content.get("processed_content", {})
        content_type = processed_content.get("content_type", "semantic")
        
        # Add user context if relevant
        if user_context.get("user_id"):
            context_parts.append("Context: User is requesting personalized information.")
        
        # Build context based on content type
        if content_type == "conversational":
            context_parts.append("This is a conversational query requiring a friendly, helpful response.")
            
        elif content_type == "structured_data":
            context_parts.append("User is asking about specific data from their account/courses.")
            structured_context = self._format_structured_content_for_context(content_data)
            if structured_context:
                context_parts.append(structured_context)
                
        elif content_type == "content_based" or content_type == "semantic":
            context_parts.append("User is asking for information that requires content-based knowledge.")
            semantic_context = self._format_semantic_content_for_context(content_data)
            if semantic_context:
                context_parts.append(semantic_context)
                
        elif content_type == "hybrid":
            context_parts.append("User query requires both personal data and content knowledge.")
            
            # Add structured data context
            if "structured_data" in content_data or "primary_content" in content_data:
                structured_context = self._format_structured_content_for_context(content_data)
                if structured_context:
                    context_parts.append("Personal/Account Data:")
                    context_parts.append(structured_context)
            
            # Add semantic content context
            semantic_context = self._format_semantic_content_for_context(content_data)
            if semantic_context:
                context_parts.append("Relevant Educational Content:")
                context_parts.append(semantic_context)
        
        # Add processing metadata as context hints
        metadata = processed_content.get("processing_metadata", {})
        if metadata.get("content_sections", 0) > 1:
            context_parts.append("Note: Response should synthesize information from multiple sources.")
        
        return "\n\n".join(context_parts)
    
    def _format_structured_content_for_context(self, content_data: Dict) -> str:
        """Format structured content for LLM context."""
        
        context_parts = []
        
        # Handle different structured content formats
        if "content_sections" in content_data:
            sections = content_data.get("content_sections", [])
            for section in sections:
                category = section.get("category", "")
                summary = section.get("summary", "")
                key_items = section.get("key_items", [])
                
                if summary:
                    context_parts.append(f"{category.title()}: {summary}")
                
                if key_items and len(key_items) <= 5:  # Avoid too much detail
                    items_text = []
                    for item in key_items:
                        if "title" in item:
                            items_text.append(f"- {item['title']}")
                        elif "metric" in item:
                            items_text.append(f"- {item['metric']}: {item['value']}")
                    if items_text:
                        context_parts.append("\n".join(items_text))
        
        elif "primary_content" in content_data:
            # Handle weighted merge format
            primary = content_data.get("primary_content", {})
            if "content_sections" in primary:
                return self._format_structured_content_for_context(primary)
        
        elif "combined_content" in content_data:
            # Handle simple combine format
            structured = content_data.get("combined_content", {}).get("structured", {})
            if structured:
                return self._format_structured_content_for_context(structured)
        
        return "\n".join(context_parts)
    
    def _format_semantic_content_for_context(self, content_data: Dict) -> str:
        """Format semantic content for LLM context."""
        
        context_parts = []
        
        # Handle different semantic content formats
        chunks = []
        
        if "content_chunks" in content_data:
            chunks = content_data.get("content_chunks", [])
        elif "semantic_content" in content_data:
            chunks = content_data.get("semantic_content", {}).get("content_chunks", [])
        elif "supporting_content" in content_data:
            chunks = content_data.get("supporting_content", {}).get("content_chunks", [])
        elif "combined_content" in content_data:
            semantic = content_data.get("combined_content", {}).get("semantic", {})
            chunks = semantic.get("content_chunks", [])
        
        # Format chunks for context
        for i, chunk in enumerate(chunks[:8]):  # Limit to avoid context overflow
            content = chunk.get("content", {})
            title = content.get("title", "")
            text = content.get("text", "")
            
            chunk_context = []
            if title:
                chunk_context.append(f"Source {i+1}: {title}")
            if text:
                # Truncate very long content
                display_text = text[:400] + "..." if len(text) > 400 else text
                chunk_context.append(display_text)
            
            if chunk_context:
                context_parts.append("\n".join(chunk_context))
                context_parts.append("---")  # Separator
        
        # Remove trailing separator
        if context_parts and context_parts[-1] == "---":
            context_parts.pop()
        
        return "\n\n".join(context_parts)
    
    def _create_synthesis_prompt(
        self,
        context: str,
        original_query: str,
        response_type: str,
        instructions: Dict
    ) -> str:
        """Create the final prompt for LLM synthesis."""
        
        base_prompt = self.response_templates.get(response_type, self.response_templates["default"])
        
        # Customize prompt based on intent signals
        intent_signals = instructions.get("intent_signals", {})
        
        # Adjust for urgency
        urgency = intent_signals.get("urgency", "normal")
        if urgency == "high":
            base_prompt += "\n\nNote: This appears to be an urgent query. Prioritize immediate, actionable information."
        
        # Adjust for content preference
        content_pref = intent_signals.get("content_preference")
        if content_pref == "summary":
            base_prompt += "\n\nNote: User prefers concise, summary-style information."
        elif content_pref == "detailed":
            base_prompt += "\n\nNote: User is looking for comprehensive, detailed information."
        
        # Adjust for scope
        scope = intent_signals.get("scope")
        if scope == "broad":
            base_prompt += "\n\nNote: Provide comprehensive coverage of the topic."
        elif scope == "narrow":
            base_prompt += "\n\nNote: Focus on specific, targeted information."
        
        return base_prompt
    
    def _initialize_response_templates(self) -> Dict[str, str]:
        """Initialize response templates for different content types."""
        
        return {
            "conversational": """
You are a helpful AI assistant for an educational platform. Respond to the user's conversational query in a friendly, helpful manner. Use the context if provided, but keep the response natural and engaging.

### Context:
{context}

### Question:
{question}

### Response:
            """,
            
            "structured_data": """
You are a helpful AI assistant. The user is asking about their personal data, course progress, or account information. Use the provided structured data to give them an accurate, helpful response. Present information clearly and highlight key points.

### Context:
{context}

### Question:
{question}

### Response:
            """,
            
            "content_based": """
You are a helpful AI assistant for educational content. Use the provided context to answer the user's question about educational topics, concepts, or course content. Provide clear, informative explanations with examples where helpful.

### Context:
{context}

### Question:
{question}

### Response:
            """,
            
            "semantic": """
You are a helpful AI assistant. Use the context below to answer the question accurately using **Markdown formatting** where appropriate. If the answer benefits from bullet points, lists, or emphasis, use them. Avoid repeating the question and provide a clean, direct answer.

### Context:
{context}

### Question:
{question}

### Response:
            """,
            
            "hybrid": """
You are a helpful AI assistant for an educational platform. The user's question requires both personal/account information and educational content knowledge. Synthesize information from both sources to provide a comprehensive, personalized response.

### Context:
{context}

### Question:
{question}

### Response:
            """,
            
            "default": """
You are a helpful AI assistant for an educational platform. Use the context below to answer the user's question accurately and helpfully. Provide clear, well-structured responses using Markdown formatting where appropriate.

### Context:
{context}

### Question:
{question}

### Response:
            """
        }
    
    def _generate_error_response(self, error_message: str) -> Generator[str, None, None]:
        """Generate error response in streaming format."""
        error_response = f"""I apologize, but I encountered an issue while processing your request: {error_message}

Please try rephrasing your question or contact support if the issue persists."""
        
        # Stream the error response word by word for consistency
        words = error_response.split()
        for word in words:
            yield word + " "
    
    def _generate_complete_response(self, context: str, query: str) -> str:
        """Generate complete response (non-streaming fallback)."""
        # This would be implemented if non-streaming response is needed
        # For now, return a basic response
        return f"Based on the available information, here's what I found regarding your query: {query[:100]}..."
    
    def optimize_response_for_streaming(self, content: str) -> Generator[str, None, None]:
        """Optimize content for smooth streaming experience."""
        
        # Split content into meaningful chunks for streaming
        sentences = self._split_into_sentences(content)
        
        for sentence in sentences:
            # Stream each sentence with slight delay simulation
            words = sentence.split()
            for word in words:
                yield word + " "
            yield "\n"  # Add newline after each sentence
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for streaming."""
        import re
        
        # Simple sentence splitting (could be enhanced with NLTK or spaCy)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def validate_response_quality(self, response_content: str, original_query: str) -> Dict[str, Any]:
        """Validate the quality of generated response."""
        
        quality_metrics = {
            "length_appropriate": True,
            "addresses_query": True,
            "coherent": True,
            "issues": []
        }
        
        # Length check
        if len(response_content) < 50:
            quality_metrics["length_appropriate"] = False
            quality_metrics["issues"].append("Response too short")
        elif len(response_content) > self.max_response_length:
            quality_metrics["length_appropriate"] = False
            quality_metrics["issues"].append("Response too long")
        
        # Query relevance check (simple keyword matching)
        query_words = set(original_query.lower().split())
        response_words = set(response_content.lower().split())
        
        if query_words and response_words:
            overlap = len(query_words & response_words) / len(query_words)
            if overlap < 0.1:  # Less than 10% overlap
                quality_metrics["addresses_query"] = False
                quality_metrics["issues"].append("Response may not address the query")
        
        # Basic coherence check
        if response_content.count('.') == 0 and len(response_content) > 100:
            quality_metrics["coherent"] = False
            quality_metrics["issues"].append("Response lacks proper sentence structure")
        
        return quality_metrics
    
    def format_response_with_metadata(
        self, 
        response_content: str, 
        processing_metadata: Dict,
        instructions: Dict
    ) -> Dict[str, Any]:
        """Format response with comprehensive metadata."""
        
        return {
            "response": response_content,
            "metadata": {
                "response_type": processing_metadata.get("content_type", "unknown"),
                "sources_used": self._extract_source_info(processing_metadata),
                "processing_chain": self._get_processing_chain_summary(instructions),
                "confidence_indicators": self._assess_response_confidence(processing_metadata),
                "suggestions": self._generate_follow_up_suggestions(response_content, instructions)
            }
        }
    
    def _extract_source_info(self, processing_metadata: Dict) -> Dict[str, Any]:
        """Extract information about sources used in the response."""
        
        input_summary = processing_metadata.get("input_summary", {})
        
        return {
            "database_results": input_summary.get("mongodb_results_count", 0),
            "content_results": input_summary.get("rag_results_count", 0),
            "total_sources": input_summary.get("mongodb_results_count", 0) + input_summary.get("rag_results_count", 0)
        }
    
    def _get_processing_chain_summary(self, instructions: Dict) -> List[str]:
        """Get summary of processing agents used."""
        
        agents_used = instructions.get("agents_to_invoke", [])
        
        agent_names = {
            "mongodb": "Database Query",
            "filter": "Content Filtering", 
            "rag": "Semantic Search",
            "post_retrieval": "Content Processing",
            "llm_synthesis": "Response Generation"
        }
        
        return [agent_names.get(agent, agent) for agent in agents_used]
    
    def _assess_response_confidence(self, processing_metadata: Dict) -> Dict[str, Any]:
        """Assess confidence in the generated response."""
        
        confidence = {
            "overall": "medium",
            "factors": []
        }
        
        input_summary = processing_metadata.get("input_summary", {})
        processing_steps = processing_metadata.get("processing_steps", {})
        
        # High confidence indicators
        if input_summary.get("mongodb_results_count", 0) > 0:
            confidence["factors"].append("Found relevant structured data")
        
        if input_summary.get("rag_results_count", 0) >= 3:
            confidence["factors"].append("Multiple relevant content sources")
            confidence["overall"] = "high"
        
        # Low confidence indicators
        if input_summary.get("total_sources", 0) == 0:
            confidence["overall"] = "low"
            confidence["factors"].append("Limited source material")
        
        if processing_steps.get("compression_applied", False):
            confidence["factors"].append("Content was compressed - some detail may be lost")
        
        return confidence
    
    def _generate_follow_up_suggestions(self, response_content: str, instructions: Dict) -> List[str]:
        """Generate follow-up suggestions based on the response."""
        
        suggestions = []
        
        # Based on response type
        response_type = instructions.get("response_type", "")
        
        if response_type == "structured_data":
            suggestions.append("Would you like more details about any specific item?")
            suggestions.append("Do you need help with next steps or actions?")
        
        elif response_type in ["content_based", "semantic"]:
            suggestions.append("Would you like more examples or practical applications?")
            suggestions.append("Are you interested in related topics?")
        
        elif response_type == "hybrid":
            suggestions.append("Would you like to explore this topic further?")
            suggestions.append("Do you need help applying this to your specific situation?")
        
        # Based on content characteristics
        if "course" in response_content.lower():
            suggestions.append("Would you like recommendations for getting started?")
        
        if len(response_content) > 500:
            suggestions.append("Would you like a summary of the key points?")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def handle_empty_context(self, original_query: str, instructions: Dict) -> Generator[str, None, None]:
        """Handle cases where no relevant context was found."""
        
        fallback_responses = {
            "course_search": "I couldn't find specific courses matching your criteria. You might want to browse our course catalog or try different search terms.",
            "progress": "I don't have access to your current progress information. Please check your dashboard or contact support.",
            "general": f"I don't have specific information about '{original_query[:50]}...' in our current database. You might want to:"
        }
        
        # Determine response type
        query_lower = original_query.lower()
        
        if any(word in query_lower for word in ["course", "learn", "study"]):
            response_key = "course_search"
        elif any(word in query_lower for word in ["progress", "complete", "finish"]):
            response_key = "progress"
        else:
            response_key = "general"
        
        response = fallback_responses[response_key]
        
        if response_key == "general":
            response += "\n\n• Try rephrasing your question\n• Check our help documentation\n• Contact support for assistance"
        
        # Stream the fallback response
        words = response.split()
        for word in words:
            yield word + " "
    
    def create_response_summary(self, full_response: str) -> str:
        """Create a brief summary of the response for metadata."""
        
        # Extract first sentence or first 100 characters
        sentences = full_response.split('.')
        if sentences and len(sentences[0]) < 150:
            return sentences[0] + "."
        else:
            return full_response[:100] + "..."
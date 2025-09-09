"""
Intent Router Agent - Step 2 of Multi-Agent Pipeline
Routes queries to appropriate processing paths based on refined intent analysis.
"""

from enum import Enum
from typing import Dict, List, Optional, Union
import logging
import re

logger = logging.getLogger(__name__)


class ProcessingPath(Enum):
    """Different processing paths for query handling."""
    CONVERSATIONAL = "conversational"  # Direct to LLM for chat responses
    OPERATIONAL = "operational"        # MongoDB → LLM
    SEMANTIC = "semantic"              # Filter → RAG → Post-Process → LLM  
    HYBRID = "hybrid"                  # MongoDB + RAG → LLM


class RouterDecision(Enum):
    """Router decision types."""
    DIRECT_RESPONSE = "direct_response"
    DATABASE_QUERY = "database_query"
    SEMANTIC_SEARCH = "semantic_search"
    HYBRID_SEARCH = "hybrid_search"


class IntentRouterAgent:
    """
    Routes refined queries to appropriate processing pipelines:
    - Conversational: Direct LLM response
    - Operational: MongoDB query + LLM formatting
    - Semantic: Vector search + LLM synthesis
    - Hybrid: Combined approach
    """
    
    def __init__(self):
        self.confidence_threshold = 0.6
        self.hybrid_threshold = 0.4  # Threshold for considering hybrid approach
        
    def route_query(self, refined_query_data: Dict) -> Dict:
        """
        Main routing logic for processed queries.
        
        Args:
            refined_query_data: Output from QueryRefinementAgent
            
        Returns:
            Routing decision with processing instructions
        """
        try:
            if "error" in refined_query_data:
                return self._create_error_route(refined_query_data["error"])
            
            query = refined_query_data["refined_query"]
            query_type = refined_query_data["query_type"]
            confidence = refined_query_data["confidence"]
            entities = refined_query_data.get("entities", [])
            intent_signals = refined_query_data.get("intent_signals", {})
            
            # Analyze routing requirements
            routing_analysis = self._analyze_routing_requirements(
                query, query_type, confidence, entities, intent_signals
            )
            
            # Make routing decision
            route_decision = self._make_routing_decision(routing_analysis)
            
            # Generate processing instructions
            processing_instructions = self._generate_processing_instructions(
                route_decision, routing_analysis, refined_query_data
            )
            
            return {
                "route_decision": route_decision.value,
                "processing_path": self._get_processing_path(route_decision).value,
                "processing_instructions": processing_instructions,
                "routing_confidence": routing_analysis["confidence"],
                "routing_metadata": routing_analysis,
                "original_query_data": refined_query_data
            }
            
        except Exception as e:
            logger.error(f"Error in intent routing: {e}")
            return self._create_error_route(f"Intent routing failed: {str(e)}")
    
    def _analyze_routing_requirements(
        self,
        query: str,
        query_type: str,
        confidence: float,
        entities: List[Dict],
        intent_signals: Dict
    ) -> Dict:
        """Analyze query to determine routing requirements."""
        
        analysis = {
            "requires_database": False,
            "requires_vector_search": False,
            "requires_user_context": False,
            "requires_real_time_data": False,
            "confidence": confidence,
            "complexity": "simple"
        }
        
        query_lower = query.lower()
        
        # Check for database requirements
        database_indicators = [
            r'my courses|my progress|my account|my profile',
            r'enrolled|completed|in progress|due date',
            r'how many|count|list all|show me all',
            r'assignment|homework|quiz|test',
            r'deadline|due|submit|grade'
        ]
        
        for indicator in database_indicators:
            if re.search(indicator, query_lower):
                analysis["requires_database"] = True
                break
        
        # Check for vector search requirements
        semantic_indicators = [
            r'explain|what is|how does|tell me about',
            r'learn about|understand|concept|definition',
            r'example|tutorial|guide|lesson content',
            r'difference between|compare|similar to'
        ]
        
        for indicator in semantic_indicators:
            if re.search(indicator, query_lower):
                analysis["requires_vector_search"] = True
                break
        
        # Check for user context requirements
        personal_indicators = [
            r'my|me|i want|i need|for me|suited for',
            r'recommend|suggest|should i|best for'
        ]
        
        for indicator in personal_indicators:
            if re.search(indicator, query_lower):
                analysis["requires_user_context"] = True
                break
        
        # Check for real-time data requirements
        realtime_indicators = [
            r'today|now|current|latest|recent',
            r'this week|this month|upcoming'
        ]
        
        for indicator in realtime_indicators:
            if re.search(indicator, query_lower):
                analysis["requires_real_time_data"] = True
                break
        
        # Assess complexity
        complexity_factors = 0
        if analysis["requires_database"]:
            complexity_factors += 1
        if analysis["requires_vector_search"]:
            complexity_factors += 1
        if analysis["requires_user_context"]:
            complexity_factors += 1
        if len(entities) > 2:
            complexity_factors += 1
        if len(intent_signals) > 2:
            complexity_factors += 1
            
        if complexity_factors >= 3:
            analysis["complexity"] = "complex"
        elif complexity_factors >= 1:
            analysis["complexity"] = "moderate"
        
        return analysis
    
    def _make_routing_decision(self, analysis: Dict) -> RouterDecision:
        """Make the primary routing decision based on analysis."""
        
        requires_db = analysis["requires_database"]
        requires_vector = analysis["requires_vector_search"]
        requires_context = analysis["requires_user_context"]
        
        # Decision matrix
        if not requires_db and not requires_vector:
            return RouterDecision.DIRECT_RESPONSE
            
        elif requires_db and requires_vector:
            return RouterDecision.HYBRID_SEARCH
            
        elif requires_db and not requires_vector:
            return RouterDecision.DATABASE_QUERY
            
        elif not requires_db and requires_vector:
            return RouterDecision.SEMANTIC_SEARCH
            
        else:
            # Fallback decision based on complexity
            if analysis["complexity"] == "complex":
                return RouterDecision.HYBRID_SEARCH
            else:
                return RouterDecision.SEMANTIC_SEARCH
    
    def _get_processing_path(self, decision: RouterDecision) -> ProcessingPath:
        """Map router decision to processing path."""
        mapping = {
            RouterDecision.DIRECT_RESPONSE: ProcessingPath.CONVERSATIONAL,
            RouterDecision.DATABASE_QUERY: ProcessingPath.OPERATIONAL,
            RouterDecision.SEMANTIC_SEARCH: ProcessingPath.SEMANTIC,
            RouterDecision.HYBRID_SEARCH: ProcessingPath.HYBRID
        }
        return mapping[decision]
    
    def _generate_processing_instructions(
        self,
        decision: RouterDecision,
        analysis: Dict,
        query_data: Dict
    ) -> Dict:
        """Generate specific processing instructions for each agent."""
        
        base_instructions = {
            "query": query_data["refined_query"],
            "original_query": query_data["original_query"],
            "entities": query_data.get("entities", []),
            "intent_signals": query_data.get("intent_signals", {}),
            "priority": self._determine_priority(query_data, analysis)
        }
        
        if decision == RouterDecision.DIRECT_RESPONSE:
            return {
                **base_instructions,
                "response_type": "conversational",
                "context_needed": "minimal",
                "agents_to_invoke": ["llm_synthesis"]
            }
            
        elif decision == RouterDecision.DATABASE_QUERY:
            return {
                **base_instructions,
                "response_type": "structured_data",
                "database_filters": self._extract_database_filters(query_data),
                "agents_to_invoke": ["mongodb", "llm_synthesis"],
                "context_needed": "user_specific"
            }
            
        elif decision == RouterDecision.SEMANTIC_SEARCH:
            return {
                **base_instructions,
                "response_type": "content_based",
                "vector_search_params": self._extract_vector_params(query_data),
                "metadata_filters": self._extract_metadata_filters_dict(query_data),  # Fixed method name
                "agents_to_invoke": ["filter", "rag", "post_retrieval", "llm_synthesis"],
                "context_needed": "content_semantic"
            }
            
        elif decision == RouterDecision.HYBRID_SEARCH:
            return {
                **base_instructions,
                "response_type": "hybrid",
                "database_filters": self._extract_database_filters(query_data),
                "vector_search_params": self._extract_vector_params(query_data),
                "metadata_filters": self._extract_metadata_filters_dict(query_data),  # Fixed method name
                "agents_to_invoke": ["mongodb", "filter", "rag", "post_retrieval", "llm_synthesis"],
                "context_needed": "comprehensive",
                "merge_strategy": self._determine_merge_strategy(analysis)
            }
            
        return base_instructions
    
    def _determine_priority(self, query_data: Dict, analysis: Dict) -> str:
        """Determine processing priority based on query characteristics."""
        urgency = query_data.get("intent_signals", {}).get("urgency", "normal")
        complexity = analysis.get("complexity", "simple")
        
        if urgency == "high" or complexity == "complex":
            return "high"
        elif urgency == "medium" or complexity == "moderate":
            return "medium"
        else:
            return "normal"
    
    def _extract_database_filters(self, query_data: Dict) -> Dict:
        """Extract filters for MongoDB queries."""
        filters = {}
        query = query_data["refined_query"].lower()
        entities = query_data.get("entities", [])
        
        # Status filters
        if re.search(r'completed|finished|done', query):
            filters["status"] = "completed"
        elif re.search(r'in progress|ongoing|current', query):
            filters["status"] = "in_progress"
        elif re.search(r'not started|upcoming|future', query):
            filters["status"] = "not_started"
        
        # Type filters
        if re.search(r'course|courses', query):
            filters["type"] = "course"
        elif re.search(r'assignment|assignments', query):
            filters["type"] = "assignment"
        elif re.search(r'quiz|quizzes|test', query):
            filters["type"] = "quiz"
        
        # Time-based filters
        if re.search(r'today|due today', query):
            filters["due_date"] = "today"
        elif re.search(r'this week|due this week', query):
            filters["due_date"] = "this_week"
        elif re.search(r'overdue|late', query):
            filters["due_date"] = "overdue"
        
        # Category filters from entities
        for entity in entities:
            if entity.get('type') in ['programming_language', 'ai_topic', 'data_topic', 'web_topic']:
                filters["category_hint"] = entity['value']
                break
        
        return filters
    
    def _extract_vector_params(self, query_data: Dict) -> Dict:
        """Extract parameters for vector search."""
        params = {
            "top_k": 10,  # Default number of results
            "search_query": query_data["refined_query"],
            "rerank": True
        }
        
        intent_signals = query_data.get("intent_signals", {})
        
        # Adjust based on scope
        scope = intent_signals.get("scope", "medium")
        if scope == "broad":
            params["top_k"] = 20
        elif scope == "narrow":
            params["top_k"] = 5
            
        # Adjust based on content preference
        content_pref = intent_signals.get("content_preference")
        if content_pref:
            params["content_type_preference"] = content_pref
            
        return params
    
    def _extract_metadata_filters_dict(self, query_data: Dict) -> Dict:
        """Extract metadata filters as a dictionary (not Qdrant format yet)."""
        filters = {}
        entities = query_data.get("entities", [])
        intent_signals = query_data.get("intent_signals", {})
        
        # Content type filters
        content_pref = intent_signals.get("content_preference")
        if content_pref:
            filters["content_type"] = content_pref
        
        # Language/topic filters from entities
        for entity in entities:
            entity_type = entity.get('type')
            if entity_type == 'programming_language':
                filters["programming_language"] = entity['value']
            elif entity_type in ['ai_topic', 'data_topic', 'web_topic']:
                filters["topic_category"] = entity['value']
            elif entity_type == 'content_type':
                filters["resource_type"] = entity['value']
        
        # Difficulty level (inferred from query patterns)
        query_lower = query_data["refined_query"].lower()
        if re.search(r'beginner|basic|introduction|intro|getting started', query_lower):
            filters["difficulty"] = "beginner"
        elif re.search(r'advanced|expert|professional|master', query_lower):
            filters["difficulty"] = "advanced"
        elif re.search(r'intermediate|mid-level', query_lower):
            filters["difficulty"] = "intermediate"
            
        return filters
    
    def _determine_merge_strategy(self, analysis: Dict) -> str:
        """Determine how to merge database and vector search results."""
        complexity = analysis.get("complexity", "simple")
        
        if complexity == "complex":
            return "weighted_synthesis"  # Sophisticated merging
        elif complexity == "moderate":
            return "contextual_merge"    # Context-aware merging
        else:
            return "simple_combine"      # Basic concatenation
    
    def _create_error_route(self, error_message: str) -> Dict:
        """Create error routing response."""
        return {
            "route_decision": RouterDecision.DIRECT_RESPONSE.value,
            "processing_path": ProcessingPath.CONVERSATIONAL.value,
            "processing_instructions": {
                "response_type": "error",
                "error_message": error_message,
                "agents_to_invoke": ["llm_synthesis"]
            },
            "routing_confidence": 0.0,
            "routing_metadata": {"error": True},
            "original_query_data": {}
        }
    
    def should_use_hybrid_approach(
        self,
        operational_confidence: float,
        semantic_confidence: float
    ) -> bool:
        """
        Determine if hybrid approach should be used based on confidence scores.
        
        Args:
            operational_confidence: Confidence for operational query handling
            semantic_confidence: Confidence for semantic query handling
            
        Returns:
            Boolean indicating whether to use hybrid approach
        """
        # Use hybrid if both approaches have reasonable confidence
        return (
            operational_confidence >= self.hybrid_threshold and
            semantic_confidence >= self.hybrid_threshold and
            abs(operational_confidence - semantic_confidence) < 0.3
        )
    
    def get_fallback_route(self, query_data: Dict) -> Dict:
        """Provide fallback routing when primary analysis fails."""
        return {
            "route_decision": RouterDecision.SEMANTIC_SEARCH.value,
            "processing_path": ProcessingPath.SEMANTIC.value,
            "processing_instructions": {
                "query": query_data.get("refined_query", ""),
                "response_type": "content_based",
                "agents_to_invoke": ["filter", "rag", "llm_synthesis"],
                "context_needed": "content_semantic",
                "fallback": True
            },
            "routing_confidence": 0.5,
            "routing_metadata": {"fallback_used": True},
            "original_query_data": query_data
        }
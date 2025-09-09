"""
Query Refinement Agent - Step 1 of Multi-Agent Pipeline
Handles query normalization, chat history injection, and basic preprocessing.
"""

import re
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Classification of query types for initial routing."""
    OPERATIONAL = "operational"  # Structured data queries
    SEMANTIC = "semantic"        # Content-based queries  
    HYBRID = "hybrid"           # Combination of both
    CONVERSATIONAL = "conversational"  # Chat/greeting


class QueryRefinementAgent:
    """
    Handles the first step of query processing:
    - Normalizes and cleans user input
    - Injects relevant chat history context
    - Performs initial query classification
    - Resolves pronouns and ambiguous references
    """
    
    def __init__(self):
        self.max_history_context = 3  # Number of previous interactions to consider
        self.min_query_length = 2
        self.max_query_length = 1000
        
    def refine_query(
        self, 
        user_query: str, 
        chat_history: Optional[List[Dict]] = None,
        user_context: Optional[Dict] = None
    ) -> Dict:
        """
        Main entry point for query refinement.
        
        Args:
            user_query: Raw user input
            chat_history: Previous conversation history
            user_context: User information (org_id, preferences, etc.)
            
        Returns:
            Dict containing refined query and metadata
        """
        try:
            # Step 1: Basic validation and cleaning
            cleaned_query = self._clean_and_validate_query(user_query)
            if not cleaned_query:
                return self._create_error_response("Invalid or empty query")
            
            # Step 2: Inject chat history context
            contextualized_query = self._inject_chat_history(
                cleaned_query, chat_history
            )
            
            # Step 3: Resolve pronouns and references
            resolved_query = self._resolve_references(
                contextualized_query, chat_history
            )
            
            # Step 4: Initial query classification
            query_type, confidence = self._classify_query(resolved_query)
            
            # Step 5: Extract key entities and intent signals
            entities = self._extract_entities(resolved_query)
            intent_signals = self._extract_intent_signals(resolved_query)
            
            return {
                "original_query": user_query,
                "refined_query": resolved_query,
                "query_type": query_type.value,
                "confidence": confidence,
                "entities": entities,
                "intent_signals": intent_signals,
                "context_used": bool(chat_history),
                "preprocessing_notes": []
            }
            
        except Exception as e:
            logger.error(f"Error in query refinement: {e}")
            return self._create_error_response(f"Query refinement failed: {str(e)}")
    
    def _clean_and_validate_query(self, query: str) -> Optional[str]:
        """Clean and validate the input query."""
        if not query or not isinstance(query, str):
            return None
            
        # Basic cleaning
        cleaned = query.strip()
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Length validation
        if len(cleaned) < self.min_query_length:
            return None
        if len(cleaned) > self.max_query_length:
            cleaned = cleaned[:self.max_query_length]
            
        return cleaned
    
    def _inject_chat_history(
        self, 
        query: str, 
        chat_history: Optional[List[Dict]]
    ) -> str:
        """Inject relevant chat history context into the query."""
        if not chat_history:
            return query
            
        # Get recent relevant history
        recent_history = self._get_relevant_history(query, chat_history)
        
        if not recent_history:
            return query
            
        # Build context string
        context_parts = []
        for interaction in recent_history:
            if interaction.get('user_query'):
                context_parts.append(f"Previous: {interaction['user_query']}")
                
        if context_parts:
            context_str = " | ".join(context_parts[-self.max_history_context:])
            return f"Context: {context_str}\nCurrent query: {query}"
            
        return query
    
    def _get_relevant_history(
        self, 
        current_query: str, 
        chat_history: List[Dict]
    ) -> List[Dict]:
        """Filter chat history for relevant context."""
        if not chat_history:
            return []
            
        # For now, return the most recent interactions
        # Future enhancement: implement semantic similarity matching
        return chat_history[-self.max_history_context:]
    
    def _resolve_references(
        self, 
        query: str, 
        chat_history: Optional[List[Dict]]
    ) -> str:
        """Resolve pronouns and ambiguous references using chat history."""
        if not chat_history:
            return query
            
        resolved_query = query
        
        # Common pronoun patterns to resolve
        pronoun_patterns = {
            r'\bit\b': self._resolve_it_reference,
            r'\bthis\b': self._resolve_this_reference,
            r'\bthat\b': self._resolve_that_reference,
            r'\bthese\b': self._resolve_these_reference,
            r'\bthose\b': self._resolve_those_reference
        }
        
        for pattern, resolver in pronoun_patterns.items():
            if re.search(pattern, resolved_query, re.IGNORECASE):
                resolved_query = resolver(resolved_query, chat_history)
                
        return resolved_query
    
    def _resolve_it_reference(self, query: str, history: List[Dict]) -> str:
        """Resolve 'it' references to previously mentioned subjects."""
        # Simple implementation - look for the most recent noun/entity
        for interaction in reversed(history):
            if interaction.get('entities'):
                for entity in interaction['entities']:
                    if entity.get('type') in ['course', 'resource', 'topic']:
                        return re.sub(
                            r'\bit\b', 
                            entity['value'], 
                            query, 
                            count=1, 
                            flags=re.IGNORECASE
                        )
        return query
    
    def _resolve_this_reference(self, query: str, history: List[Dict]) -> str:
        """Resolve 'this' references."""
        # Similar logic to 'it' resolution
        return self._resolve_it_reference(query, history)
    
    def _resolve_that_reference(self, query: str, history: List[Dict]) -> str:
        """Resolve 'that' references."""
        return self._resolve_it_reference(query, history)
    
    def _resolve_these_reference(self, query: str, history: List[Dict]) -> str:
        """Resolve 'these' references to plural entities."""
        return query  # Placeholder for plural resolution
    
    def _resolve_those_reference(self, query: str, history: List[Dict]) -> str:
        """Resolve 'those' references to plural entities."""
        return query  # Placeholder for plural resolution
    
    def _classify_query(self, query: str) -> Tuple[QueryType, float]:
        """
        Perform initial query classification.
        
        Returns:
            Tuple of (QueryType, confidence_score)
        """
        query_lower = query.lower()
        
        # Conversational patterns
        conversational_patterns = [
            r'hello|hi|hey|good morning|good afternoon|good evening',
            r'thank you|thanks|bye|goodbye|see you',
            r'how are you|what\'s up|how\'s it going'
        ]
        
        for pattern in conversational_patterns:
            if re.search(pattern, query_lower):
                return QueryType.CONVERSATIONAL, 0.9
        
        # Operational patterns (structured data queries)
        operational_patterns = [
            r'how many|count|number of|list all|show me all',
            r'my courses|my progress|my assignments|my account',
            r'due date|deadline|when is|what time',
            r'enrolled|completed|in progress|started'
        ]
        
        operational_score = 0.0
        for pattern in operational_patterns:
            if re.search(pattern, query_lower):
                operational_score += 0.3
                
        # Semantic patterns (content-based queries)
        semantic_patterns = [
            r'explain|what is|how does|tell me about|describe',
            r'learn about|understand|concept|definition',
            r'example|tutorial|guide|lesson',
            r'difference between|compare|contrast'
        ]
        
        semantic_score = 0.0
        for pattern in semantic_patterns:
            if re.search(pattern, query_lower):
                semantic_score += 0.3
        
        # Hybrid indicators
        hybrid_patterns = [
            r'recommend|suggest|best.*for me|should I',
            r'my.*about|for my.*level|suited for'
        ]
        
        hybrid_score = 0.0
        for pattern in hybrid_patterns:
            if re.search(pattern, query_lower):
                hybrid_score += 0.4
        
        # Determine primary classification
        scores = {
            QueryType.OPERATIONAL: operational_score,
            QueryType.SEMANTIC: semantic_score,
            QueryType.HYBRID: hybrid_score
        }
        
        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type], 1.0)
        
        # Default to semantic if no clear classification
        if confidence < 0.3:
            return QueryType.SEMANTIC, 0.5
            
        return best_type, confidence
    
    def _extract_entities(self, query: str) -> List[Dict]:
        """Extract named entities and key terms from the query."""
        entities = []
        query_lower = query.lower()
        
        # Course-related entities
        course_patterns = {
            r'python|javascript|java|c\+\+|html|css|sql': 'programming_language',
            r'machine learning|ai|artificial intelligence|ml': 'ai_topic',
            r'data science|analytics|statistics': 'data_topic',
            r'web development|frontend|backend|fullstack': 'web_topic',
            r'course|lesson|module|chapter|tutorial': 'content_type'
        }
        
        for pattern, entity_type in course_patterns.items():
            matches = re.findall(pattern, query_lower)
            for match in matches:
                entities.append({
                    'value': match,
                    'type': entity_type,
                    'confidence': 0.8
                })
        
        # Time-related entities
        time_patterns = {
            r'today|tomorrow|yesterday': 'time_relative',
            r'this week|next week|last week': 'time_week',
            r'this month|next month': 'time_month'
        }
        
        for pattern, entity_type in time_patterns.items():
            matches = re.findall(pattern, query_lower)
            for match in matches:
                entities.append({
                    'value': match,
                    'type': entity_type,
                    'confidence': 0.9
                })
        
        return entities
    
    def _extract_intent_signals(self, query: str) -> Dict:
        """Extract signals that help determine user intent."""
        signals = {
            'urgency': self._detect_urgency(query),
            'scope': self._detect_scope(query),
            'action_type': self._detect_action_type(query),
            'content_preference': self._detect_content_preference(query)
        }
        
        return {k: v for k, v in signals.items() if v is not None}
    
    def _detect_urgency(self, query: str) -> Optional[str]:
        """Detect urgency indicators in the query."""
        query_lower = query.lower()
        
        if re.search(r'urgent|asap|immediately|right now|quickly', query_lower):
            return 'high'
        elif re.search(r'soon|today|by.*end of', query_lower):
            return 'medium'
        else:
            return 'normal'
    
    def _detect_scope(self, query: str) -> Optional[str]:
        """Detect the scope of the query (specific vs general)."""
        query_lower = query.lower()
        
        if re.search(r'all|every|complete|entire|comprehensive', query_lower):
            return 'broad'
        elif re.search(r'specific|particular|exact|just|only', query_lower):
            return 'narrow'
        else:
            return 'medium'
    
    def _detect_action_type(self, query: str) -> Optional[str]:
        """Detect the type of action the user wants to perform."""
        query_lower = query.lower()
        
        action_patterns = {
            r'find|search|look for|locate': 'search',
            r'learn|study|understand|master': 'learn',
            r'create|make|build|develop': 'create',
            r'review|check|verify|validate': 'review',
            r'compare|contrast|difference': 'compare',
            r'recommend|suggest|advise': 'recommend'
        }
        
        for pattern, action in action_patterns.items():
            if re.search(pattern, query_lower):
                return action
                
        return None
    
    def _detect_content_preference(self, query: str) -> Optional[str]:
        """Detect user's content format preferences."""
        query_lower = query.lower()
        
        if re.search(r'video|watch|visual|demonstration', query_lower):
            return 'video'
        elif re.search(r'read|text|article|document', query_lower):
            return 'text'
        elif re.search(r'hands.on|practice|exercise|lab', query_lower):
            return 'interactive'
        elif re.search(r'summary|brief|overview|quick', query_lower):
            return 'summary'
            
        return None
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create standardized error response."""
        return {
            "original_query": "",
            "refined_query": "",
            "query_type": QueryType.SEMANTIC.value,
            "confidence": 0.0,
            "entities": [],
            "intent_signals": {},
            "context_used": False,
            "preprocessing_notes": [],
            "error": error_message
        }
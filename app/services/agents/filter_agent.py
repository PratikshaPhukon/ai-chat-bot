"""
Filter Agent - Step 3b of Multi-Agent Pipeline
Extracts metadata filters from queries for Qdrant vector search optimization.
"""

from typing import Dict, List, Optional, Any
import re
import logging

logger = logging.getLogger(__name__)


class FilterAgent:
    """
    Extracts and builds metadata filters for Qdrant vector search.
    Reduces irrelevant matches and improves search precision.
    """
    
    def __init__(self):
        # Predefined categories and their variations
        self.category_mappings = {
            "programming": ["programming", "coding", "development", "software"],
            "data_science": ["data science", "analytics", "statistics", "ml", "machine learning"],
            "web_development": ["web", "frontend", "backend", "fullstack", "html", "css", "javascript"],
            "artificial_intelligence": ["ai", "artificial intelligence", "neural networks", "deep learning"],
            "business": ["business", "management", "entrepreneurship", "marketing", "finance"],
            "design": ["design", "ui", "ux", "graphic", "visual", "creative"],
            "mathematics": ["math", "mathematics", "calculus", "algebra", "geometry"],
            "science": ["science", "physics", "chemistry", "biology", "research"]
        }
        
        self.difficulty_keywords = {
            "beginner": ["beginner", "basic", "introduction", "intro", "getting started", "fundamentals", "101"],
            "intermediate": ["intermediate", "mid-level", "advanced beginner", "next step", "building on"],
            "advanced": ["advanced", "expert", "professional", "master", "complex", "in-depth", "specialized"]
        }
        
        self.content_type_mappings = {
            "video": ["video", "watch", "visual", "demonstration", "tutorial video"],
            "text": ["text", "article", "reading", "document", "book", "written"],
            "interactive": ["hands-on", "practice", "exercise", "lab", "interactive", "workshop"],
            "quiz": ["quiz", "test", "assessment", "exam", "evaluation"],
            "assignment": ["assignment", "homework", "project", "task"]
        }
    
    def extract_filters(self, instructions: Dict) -> Dict:
        """
        Extract metadata filters from processing instructions.
        
        Args:
            instructions: Processing instructions from IntentRouterAgent
            
        Returns:
            Qdrant-compatible filter payload
        """
        try:
            query = instructions.get("query", "")
            entities = instructions.get("entities", [])
            intent_signals = instructions.get("intent_signals", {})
            metadata_filters = instructions.get("metadata_filters", {})
            
            # Build filter conditions
            filter_conditions = []
            
            # Extract category filters
            category_condition = self._extract_category_filters(query, entities)
            if category_condition:
                filter_conditions.append(category_condition)
            
            # Extract difficulty filters
            difficulty_condition = self._extract_difficulty_filters(query, intent_signals)
            if difficulty_condition:
                filter_conditions.append(difficulty_condition)
            
            # Extract content type filters
            content_type_condition = self._extract_content_type_filters(query, intent_signals)
            if content_type_condition:
                filter_conditions.append(content_type_condition)
            
            # Extract language filters
            language_condition = self._extract_language_filters(query, entities)
            if language_condition:
                filter_conditions.append(language_condition)
            
            # Extract temporal filters
            temporal_condition = self._extract_temporal_filters(query, entities)
            if temporal_condition:
                filter_conditions.append(temporal_condition)
            
            # Build final filter payload in correct Qdrant format
            if filter_conditions:
                if len(filter_conditions) == 1:
                    qdrant_filter = filter_conditions[0]
                else:
                    qdrant_filter = {"must": filter_conditions}
            else:
                qdrant_filter = None
            
            # Apply any pre-computed metadata filters
            if metadata_filters and qdrant_filter:
                qdrant_filter = self._merge_filters(qdrant_filter, metadata_filters)
            elif metadata_filters and not qdrant_filter:
                qdrant_filter = self._convert_metadata_to_qdrant_format(metadata_filters)
            
            return {
                "qdrant_filter": qdrant_filter,
                "filter_summary": self._create_filter_summary(qdrant_filter),
                "extracted_successfully": True
            }
            
        except Exception as e:
            logger.error(f"Error extracting filters: {e}")
            return {
                "qdrant_filter": None,
                "filter_summary": {"error": str(e)},
                "extracted_successfully": False
            }
    
    def _extract_category_filters(self, query: str, entities: List[Dict]) -> Optional[Dict]:
        """Extract category-based filters in Qdrant format."""
        query_lower = query.lower()
        
        # Check entities first
        for entity in entities:
            entity_type = entity.get('type')
            entity_value = entity.get('value', '').lower()
            
            if entity_type in ['programming_language', 'ai_topic', 'data_topic', 'web_topic']:
                # Map to broader categories
                for category, keywords in self.category_mappings.items():
                    if any(keyword in entity_value for keyword in keywords):
                        return {
                            "key": "category",
                            "match": {"value": category}
                        }
        
        # Check query text for category keywords
        for category, keywords in self.category_mappings.items():
            if any(keyword in query_lower for keyword in keywords):
                return {
                    "key": "category",
                    "match": {"value": category}
                }
        
        return None
    
    def _extract_difficulty_filters(self, query: str, intent_signals: Dict) -> Optional[Dict]:
        """Extract difficulty level filters in Qdrant format."""
        query_lower = query.lower()
        
        for difficulty, keywords in self.difficulty_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return {
                    "key": "difficulty",
                    "match": {"value": difficulty}
                }
        
        return None
    
    def _extract_content_type_filters(self, query: str, intent_signals: Dict) -> Optional[Dict]:
        """Extract content type preferences in Qdrant format."""
        query_lower = query.lower()
        
        # Check intent signals first
        content_pref = intent_signals.get("content_preference")
        if content_pref and content_pref in self.content_type_mappings:
            return {
                "key": "content_type",
                "match": {"value": content_pref}
            }
        
        # Check query text
        for content_type, keywords in self.content_type_mappings.items():
            if any(keyword in query_lower for keyword in keywords):
                return {
                    "key": "content_type", 
                    "match": {"value": content_type}
                }
        
        return None
    
    def _extract_language_filters(self, query: str, entities: List[Dict]) -> Optional[Dict]:
        """Extract programming language or natural language filters in Qdrant format."""
        
        # Programming languages from entities
        for entity in entities:
            if entity.get('type') == 'programming_language':
                return {
                    "key": "programming_language",
                    "match": {"value": entity['value']}
                }
        
        # Natural language preferences
        query_lower = query.lower()
        if re.search(r'\ben\b|english', query_lower):
            return {
                "key": "language",
                "match": {"value": "en"}
            }
        elif re.search(r'\bhi\b|hindi', query_lower):
            return {
                "key": "language", 
                "match": {"value": "hi"}
            }
        
        return None
    
    def _extract_temporal_filters(self, query: str, entities: List[Dict]) -> Optional[Dict]:
        """Extract time-based filters in Qdrant format."""
        query_lower = query.lower()
        
        # Look for time-based entities
        for entity in entities:
            entity_type = entity.get('type')
            if entity_type in ['time_relative', 'time_week', 'time_month']:
                # Convert to date range filter
                if 'recent' in entity['value'] or 'latest' in query_lower:
                    return {
                        "key": "created_at",
                        "range": {
                            "gte": "now-30d"  # Last 30 days
                        }
                    }
        
        return None
    
    def _convert_metadata_to_qdrant_format(self, metadata_filters: Dict) -> Dict:
        """Convert metadata filters to proper Qdrant format."""
        conditions = []
        
        for key, value in metadata_filters.items():
            if key in ["content_type", "programming_language", "topic_category", "difficulty", "language"]:
                conditions.append({
                    "key": key,
                    "match": {"value": value}
                })
            elif key == "resource_type":
                conditions.append({
                    "key": "type",
                    "match": {"value": value}
                })
        
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"must": conditions}
        else:
            return None
    
    def _merge_filters(self, base_filter: Dict, additional_filters: Dict) -> Dict:
        """Merge additional filters with base filters in Qdrant format."""
        
        # Convert additional filters to Qdrant format first
        additional_qdrant = self._convert_metadata_to_qdrant_format(additional_filters)
        if not additional_qdrant:
            return base_filter
        
        # If base_filter has "must" conditions
        if "must" in base_filter:
            base_conditions = base_filter["must"]
            if "must" in additional_qdrant:
                base_conditions.extend(additional_qdrant["must"])
            else:
                base_conditions.append(additional_qdrant)
            return {"must": base_conditions}
        
        # If additional has "must" conditions
        elif "must" in additional_qdrant:
            return {"must": [base_filter] + additional_qdrant["must"]}
        
        # Both are single conditions
        else:
            return {"must": [base_filter, additional_qdrant]}
    
    def _create_filter_summary(self, filter_payload: Optional[Dict]) -> Dict:
        """Create human-readable summary of applied filters."""
        summary = {
            "total_filters": 0,
            "filter_types": [],
            "description": []
        }
        
        if not filter_payload:
            return summary
        
        conditions = []
        
        if "must" in filter_payload:
            conditions = filter_payload["must"]
        elif "key" in filter_payload:
            conditions = [filter_payload]
        
        summary["total_filters"] = len(conditions)
        
        # Analyze conditions
        for condition in conditions:
            if "key" in condition:
                key = condition["key"]
                summary["filter_types"].append(key)
                
                if "match" in condition and "value" in condition["match"]:
                    value = condition["match"]["value"]
                    
                    if key == "category":
                        summary["description"].append(f"Category: {value}")
                    elif key == "difficulty":
                        summary["description"].append(f"Difficulty: {value}")
                    elif key == "content_type":
                        summary["description"].append(f"Content type: {value}")
                    elif key == "programming_language":
                        summary["description"].append(f"Programming language: {value}")
                    elif key == "language":
                        summary["description"].append(f"Language: {value}")
        
        return summary
    
    def validate_filters(self, filter_payload: Optional[Dict]) -> Dict:
        """Validate the generated filter payload for Qdrant compatibility."""
        validation_result = {
            "valid": True,
            "issues": [],
            "recommendations": []
        }
        
        if not filter_payload:
            return validation_result
        
        try:
            # Check if it's a single condition or has must/should structure
            if "key" in filter_payload:
                # Single condition format
                if "match" not in filter_payload and "range" not in filter_payload:
                    validation_result["valid"] = False
                    validation_result["issues"].append("Single condition must have 'match' or 'range' field")
                    
            elif "must" in filter_payload:
                # Must conditions format
                must_conditions = filter_payload["must"]
                if not isinstance(must_conditions, list):
                    validation_result["valid"] = False
                    validation_result["issues"].append("'must' field must be a list")
                else:
                    for i, condition in enumerate(must_conditions):
                        if not isinstance(condition, dict):
                            validation_result["issues"].append(f"Condition {i} must be a dictionary")
                            validation_result["valid"] = False
                        elif "key" not in condition:
                            validation_result["issues"].append(f"Condition {i} missing 'key' field")
                            validation_result["valid"] = False
                        elif "match" not in condition and "range" not in condition:
                            validation_result["issues"].append(f"Condition {i} must have 'match' or 'range' field")
                            validation_result["valid"] = False
            else:
                validation_result["valid"] = False
                validation_result["issues"].append("Filter must have either 'key' for single condition or 'must'/'should' for multiple conditions")
            
            # Check for overly restrictive filters
            condition_count = 0
            if "must" in filter_payload:
                condition_count = len(filter_payload["must"])
            elif "key" in filter_payload:
                condition_count = 1
                
            if condition_count > 5:
                validation_result["recommendations"].append("Consider reducing filters - too many may limit results")
        
        except Exception as e:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def optimize_filters(self, filter_payload: Optional[Dict], expected_result_count: int = 10) -> Optional[Dict]:
        """Optimize filters based on expected result requirements."""
        if not filter_payload:
            return None
            
        # If single condition, return as is
        if "key" in filter_payload:
            return filter_payload
            
        # If multiple conditions, potentially convert some to 'should' conditions
        if "must" in filter_payload:
            must_conditions = filter_payload["must"]
            
            if len(must_conditions) > 3:
                # Priority order: category > difficulty > content_type > language
                priority_keys = ["category", "difficulty", "content_type", "programming_language", "language"]
                
                high_priority = []
                low_priority = []
                
                for condition in must_conditions:
                    key = condition.get("key", "")
                    if key in priority_keys[:2]:  # Keep top 2 priority types
                        high_priority.append(condition)
                    else:
                        low_priority.append(condition)
                
                if high_priority and low_priority:
                    return {
                        "must": high_priority,
                        "should": low_priority,
                        "minimum_should_match": 1
                    }
        
        return filter_payload
    
    def get_filter_explanation(self, filter_payload: Optional[Dict]) -> str:
        """Generate human-readable explanation of applied filters."""
        if not filter_payload:
            return "No specific filters applied - searching all content."
        
        explanations = []
        conditions = []
        
        if "must" in filter_payload:
            conditions = filter_payload["must"]
        elif "key" in filter_payload:
            conditions = [filter_payload]
        
        for condition in conditions:
            key = condition.get("key", "")
            
            if "match" in condition and "value" in condition["match"]:
                value = condition["match"]["value"]
                
                if key == "category":
                    explanations.append(f"focusing on {value} content")
                elif key == "difficulty":
                    explanations.append(f"filtering for {value} level")
                elif key == "content_type":
                    explanations.append(f"looking for {value} materials")
                elif key == "programming_language":
                    explanations.append(f"related to {value}")
                elif key == "language":
                    explanations.append(f"in {value} language")
            
            elif "range" in condition:
                if key == "created_at":
                    explanations.append("prioritizing recent content")
        
        if explanations:
            return f"Search refined by: {', '.join(explanations)}."
        else:
            return "Custom filters applied to improve search relevance."
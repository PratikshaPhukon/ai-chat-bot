"""
RAG Agent - Step 4 of Multi-Agent Pipeline
Performs filtered vector search in Qdrant for semantic content retrieval.
"""

from typing import Dict, List, Optional, Any
import logging
from app.helpers.qdrant_helper import search_qdrant
from app.helpers.vertex_helper import get_query_vector

logger = logging.getLogger(__name__)


class RAGAgent:
    """
    Handles Retrieval-Augmented Generation (RAG) operations:
    - Generates embeddings for semantic queries
    - Performs filtered vector search in Qdrant
    - Retrieves relevant content chunks with metadata
    """
    
    def __init__(self):
        self.default_top_k = 10
        self.max_top_k = 50
        self.min_similarity_score = 0.7
        self.embedding_model = "gemini-embedding-001"  # 3072 dimensions
        
    async def retrieve_content(
        self, 
        instructions: Dict, 
        filter_payload: Dict,
        user_context: Dict
    ) -> Dict:
        """
        Main content retrieval function.
        
        Args:
            instructions: Processing instructions from IntentRouterAgent
            filter_payload: Qdrant filters from FilterAgent
            user_context: User context (org_id, preferences, etc.)
            
        Returns:
            Retrieved content with metadata
        """
        try:
            org_id = user_context.get("org_id")
            if not org_id:
                return self._create_error_response("Missing organization context")
            
            # Extract search parameters
            search_params = self._extract_search_parameters(instructions)
            query_text = search_params["query_text"]
            top_k = search_params["top_k"]
            
            # Generate query embedding
            try:
                query_vector = get_query_vector(query_text)
                logger.info(f"Generated embedding vector with {len(query_vector)} dimensions")
            except Exception as e:
                logger.error(f"Failed to generate query embedding: {e}")
                return self._create_error_response(f"Embedding generation failed: {str(e)}")
            
            # Prepare Qdrant filter - extract from filter_payload
            qdrant_filter = None
            if filter_payload and filter_payload.get("extracted_successfully", False):
                qdrant_filter = filter_payload.get("qdrant_filter")
                logger.info(f"Using Qdrant filter: {qdrant_filter}")
            else:
                logger.info("No filters applied - searching all content")
            
            # Get token credentials from user context
            token_credentials = user_context.get("token_credentials")
            if not token_credentials:
                return self._create_error_response("Missing authentication credentials")
            
            # Perform vector search
            try:
                search_results = search_qdrant(
                    query_vector=query_vector,
                    token_credentials=token_credentials,
                    collection_name=org_id,
                    top=top_k,
                    qdrant_filter=qdrant_filter
                )
                logger.info(f"Qdrant search returned {len(search_results)} results")
            except Exception as e:
                logger.error(f"Qdrant search failed: {e}")
                return self._create_error_response(f"Vector search failed: {str(e)}")
            
            if not search_results:
                return self._create_empty_response("No relevant content found")
            
            # Process and enrich results
            processed_results = self._process_search_results(
                search_results, 
                query_text, 
                instructions
            )
            
            # Apply additional filtering and ranking
            filtered_results = self._apply_post_search_filtering(
                processed_results,
                instructions,
                user_context
            )
            
            return {
                "success": True,
                "results": filtered_results,
                "total_found": len(search_results),
                "returned_count": len(filtered_results),
                "search_metadata": {
                    "query_embedding_dims": len(query_vector),
                    "filters_applied": qdrant_filter,
                    "search_params": search_params,
                    "min_score": min([r.get("score", 0) for r in filtered_results]) if filtered_results else 0,
                    "max_score": max([r.get("score", 0) for r in filtered_results]) if filtered_results else 0
                }
            }
            
        except Exception as e:
            logger.error(f"RAG content retrieval failed: {e}")
            return self._create_error_response(f"Content retrieval failed: {str(e)}")
    
    def _extract_search_parameters(self, instructions: Dict) -> Dict:
        """Extract and validate search parameters from instructions."""
        vector_params = instructions.get("vector_search_params", {})
        
        # Extract query text
        query_text = instructions.get("query", "")
        if not query_text:
            query_text = instructions.get("original_query", "")
        
        # Extract top_k with bounds checking
        top_k = vector_params.get("top_k", self.default_top_k)
        top_k = max(1, min(top_k, self.max_top_k))
        
        # Extract other parameters
        rerank = vector_params.get("rerank", True)
        content_type_preference = vector_params.get("content_type_preference")
        
        return {
            "query_text": query_text,
            "top_k": top_k,
            "rerank": rerank,
            "content_type_preference": content_type_preference
        }
    
    def _process_search_results(
        self, 
        search_results: List[Dict], 
        query_text: str,
        instructions: Dict
    ) -> List[Dict]:
        """Process and enrich search results with additional metadata."""
        processed_results = []
        
        for result in search_results:
            try:
                # Extract basic information
                payload = result.get("payload", {})
                score = result.get("score", 0.0)
                
                # Enrich with processing metadata
                processed_result = {
                    "id": result.get("id", ""),
                    "score": score,
                    "content": {
                        "title": payload.get("title", ""),
                        "text": payload.get("chunk_text", payload.get("text", "")),
                        "resource_id": payload.get("resource_id", ""),
                        "course_id": payload.get("course_id", ""),
                        "content_type": payload.get("content_type", ""),
                        "category": payload.get("category", "")
                    },
                    "metadata": {
                        "chunk_index": payload.get("chunk_index", 0),
                        "chunk_length": payload.get("chunk_length", 0),
                        "language": payload.get("language", "en"),
                        "created_at": payload.get("created_at"),
                        "resource_type": payload.get("type", "")
                    },
                    "relevance": self._calculate_relevance_score(payload, query_text, score)
                }
                
                processed_results.append(processed_result)
                
            except Exception as e:
                logger.warning(f"Error processing search result: {e}")
                continue
        
        return processed_results
    
    def _calculate_relevance_score(
        self, 
        payload: Dict, 
        query_text: str, 
        vector_score: float
    ) -> Dict:
        """Calculate comprehensive relevance score."""
        relevance = {
            "vector_similarity": vector_score,
            "text_match": 0.0,
            "title_match": 0.0,
            "category_match": 0.0,
            "overall": vector_score
        }
        
        query_lower = query_text.lower()
        
        # Text content matching
        content_text = payload.get("chunk_text", "").lower()
        if content_text:
            # Simple keyword matching (could be enhanced with fuzzy matching)
            query_words = set(query_lower.split())
            content_words = set(content_text.split())
            if query_words and content_words:
                relevance["text_match"] = len(query_words & content_words) / len(query_words)
        
        # Title matching
        title = payload.get("title", "").lower()
        if title:
            query_words = set(query_lower.split())
            title_words = set(title.split())
            if query_words and title_words:
                relevance["title_match"] = len(query_words & title_words) / len(query_words)
        
        # Category matching
        category = payload.get("category", "").lower()
        if category and any(cat_word in query_lower for cat_word in category.split()):
            relevance["category_match"] = 0.5
        
        # Calculate overall relevance (weighted combination)
        relevance["overall"] = (
            0.6 * relevance["vector_similarity"] +
            0.2 * relevance["text_match"] +
            0.15 * relevance["title_match"] +
            0.05 * relevance["category_match"]
        )
        
        return relevance
    
    def _apply_post_search_filtering(
        self,
        results: List[Dict],
        instructions: Dict,
        user_context: Dict
    ) -> List[Dict]:
        """Apply additional filtering and ranking after vector search."""
        filtered_results = results.copy()
        
        # Filter by minimum relevance score
        filtered_results = [
            r for r in filtered_results 
            if r["relevance"]["overall"] >= self.min_similarity_score
        ]
        
        # Apply content type preferences
        search_params = instructions.get("vector_search_params", {})
        content_pref = search_params.get("content_type_preference")
        if content_pref:
            # Boost results matching content preference
            for result in filtered_results:
                if result["content"]["content_type"] == content_pref:
                    result["relevance"]["overall"] += 0.1
        
        # Apply user preferences if available
        user_prefs = user_context.get("preferences", {})
        if user_prefs:
            # Language preference
            preferred_lang = user_prefs.get("language", "en")
            for result in filtered_results:
                if result["metadata"]["language"] == preferred_lang:
                    result["relevance"]["overall"] += 0.05
        
        # Sort by overall relevance
        filtered_results.sort(
            key=lambda x: x["relevance"]["overall"], 
            reverse=True
        )
        
        # Remove duplicates based on content similarity
        filtered_results = self._remove_duplicate_content(filtered_results)
        
        return filtered_results
    
    def _remove_duplicate_content(self, results: List[Dict]) -> List[Dict]:
        """Remove results with very similar content to avoid redundancy."""
        if len(results) <= 1:
            return results
        
        unique_results = []
        seen_content = set()
        
        for result in results:
            # Create content hash for similarity detection
            content_text = result["content"]["text"][:200]  # First 200 chars
            resource_id = result["content"]["resource_id"]
            
            # Simple deduplication key
            content_key = f"{resource_id}_{hash(content_text) % 10000}"
            
            if content_key not in seen_content:
                unique_results.append(result)
                seen_content.add(content_key)
            
            # Limit to prevent excessive results
            if len(unique_results) >= 20:
                break
        
        return unique_results
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create standardized error response."""
        return {
            "success": False,
            "results": [],
            "total_found": 0,
            "returned_count": 0,
            "error": error_message,
            "search_metadata": {}
        }
    
    def _create_empty_response(self, message: str) -> Dict:
        """Create response for empty search results."""
        return {
            "success": True,
            "results": [],
            "total_found": 0,
            "returned_count": 0,
            "message": message,
            "search_metadata": {}
        }
    
    def get_search_statistics(self, results: Dict) -> Dict:
        """Generate search statistics for monitoring and optimization."""
        if not results.get("success"):
            return {"error": "No valid results to analyze"}
        
        search_results = results.get("results", [])
        
        if not search_results:
            return {"empty_results": True}
        
        # Calculate statistics
        scores = [r["relevance"]["overall"] for r in search_results]
        vector_scores = [r["relevance"]["vector_similarity"] for r in search_results]
        
        stats = {
            "result_count": len(search_results),
            "score_stats": {
                "min_overall": min(scores),
                "max_overall": max(scores),
                "avg_overall": sum(scores) / len(scores),
                "min_vector": min(vector_scores),
                "max_vector": max(vector_scores),
                "avg_vector": sum(vector_scores) / len(vector_scores)
            },
            "content_distribution": self._analyze_content_distribution(search_results),
            "quality_indicators": {
                "high_relevance_count": len([s for s in scores if s >= 0.8]),
                "medium_relevance_count": len([s for s in scores if 0.6 <= s < 0.8]),
                "low_relevance_count": len([s for s in scores if s < 0.6])
            }
        }
        
        return stats
    
    def _analyze_content_distribution(self, results: List[Dict]) -> Dict:
        """Analyze the distribution of content types in results."""
        distribution = {
            "content_types": {},
            "categories": {},
            "languages": {},
            "resource_types": {}
        }
        
        for result in results:
            content = result.get("content", {})
            metadata = result.get("metadata", {})
            
            # Content type distribution
            content_type = content.get("content_type", "unknown")
            distribution["content_types"][content_type] = distribution["content_types"].get(content_type, 0) + 1
            
            # Category distribution
            category = content.get("category", "unknown")
            distribution["categories"][category] = distribution["categories"].get(category, 0) + 1
            
            # Language distribution
            language = metadata.get("language", "unknown")
            distribution["languages"][language] = distribution["languages"].get(language, 0) + 1
            
            # Resource type distribution
            resource_type = metadata.get("resource_type", "unknown")
            distribution["resource_types"][resource_type] = distribution["resource_types"].get(resource_type, 0) + 1
        
        return distribution
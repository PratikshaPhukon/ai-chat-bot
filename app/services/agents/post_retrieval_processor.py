"""
Post-Retrieval Processor - Step 5 of Multi-Agent Pipeline
Re-ranks and compresses retrieved content for optimal LLM input.
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import re
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class PostRetrievalProcessor:
    """
    Processes retrieved content for final LLM consumption:
    - Re-ranks results based on query relevance and quality
    - Compresses content to key points
    - Removes redundancy and noise
    - Optimizes for LLM context window
    """
    
    def __init__(self):
        self.max_context_length = 8000  # Characters for LLM context
        self.max_chunks = 15  # Maximum number of chunks to process
        self.min_chunk_quality_score = 0.3
        self.compression_ratio = 0.7  # Target compression ratio
        
    def process_retrieved_content(
        self,
        rag_results: Dict,
        mongodb_results: Optional[Dict],
        instructions: Dict
    ) -> Dict:
        """
        Main processing function for retrieved content.
        
        Args:
            rag_results: Results from RAGAgent
            mongodb_results: Results from MongoDBAgent (if hybrid)
            instructions: Processing instructions from IntentRouterAgent
            
        Returns:
            Processed and optimized content for LLM synthesis
        """
        try:
            # Extract and validate input
            rag_content = rag_results.get("results", []) if rag_results.get("success") else []
            mongodb_content = mongodb_results.get("results", []) if mongodb_results and mongodb_results.get("success") else []
            
            processing_type = instructions.get("response_type", "content_based")
            
            # Process based on content type
            if processing_type == "hybrid":
                processed_content = self._process_hybrid_content(
                    rag_content, mongodb_content, instructions
                )
            elif processing_type == "structured_data":
                processed_content = self._process_structured_content(
                    mongodb_content, instructions
                )
            else:  # content_based or semantic
                processed_content = self._process_semantic_content(
                    rag_content, instructions
                )
            
            # Apply final optimizations
            optimized_content = self._optimize_for_llm(processed_content, instructions)
            
            # Generate processing metadata
            metadata = self._generate_processing_metadata(
                rag_results, mongodb_results, processed_content, optimized_content
            )
            
            return {
                "success": True,
                "processed_content": optimized_content,
                "content_type": processing_type,
                "processing_metadata": metadata,
                "ready_for_synthesis": True
            }
            
        except Exception as e:
            logger.error(f"Post-retrieval processing failed: {e}")
            return self._create_error_response(f"Content processing failed: {str(e)}")
    
    def _process_hybrid_content(
        self,
        rag_content: List[Dict],
        mongodb_content: List[Dict],
        instructions: Dict
    ) -> Dict:
        """Process hybrid content combining structured and semantic results."""
        
        # Separate and process each content type
        structured_summary = self._summarize_structured_data(mongodb_content)
        semantic_chunks = self._process_and_rank_semantic_content(rag_content, instructions)
        
        # Determine merge strategy
        merge_strategy = instructions.get("merge_strategy", "contextual_merge")
        
        if merge_strategy == "weighted_synthesis":
            merged_content = self._weighted_merge(structured_summary, semantic_chunks, instructions)
        elif merge_strategy == "contextual_merge":
            merged_content = self._contextual_merge(structured_summary, semantic_chunks, instructions)
        else:  # simple_combine
            merged_content = self._simple_combine(structured_summary, semantic_chunks)
        
        return merged_content
    
    def _process_structured_content(
        self,
        mongodb_content: List[Dict],
        instructions: Dict
    ) -> Dict:
        """Process structured database results."""
        
        if not mongodb_content:
            return {
                "content_sections": [],
                "summary": "No structured data found matching your query.",
                "data_points": [],
                "total_items": 0
            }
        
        # Categorize structured content
        categorized_content = self._categorize_structured_content(mongodb_content)
        
        # Generate summaries for each category
        content_sections = []
        for category, items in categorized_content.items():
            section = self._create_structured_section(category, items, instructions)
            content_sections.append(section)
        
        # Create overall summary
        summary = self._create_structured_summary(mongodb_content, categorized_content)
        
        # Extract key data points
        data_points = self._extract_key_data_points(mongodb_content)
        
        return {
            "content_sections": content_sections,
            "summary": summary,
            "data_points": data_points,
            "total_items": len(mongodb_content)
        }
    
    def _process_semantic_content(
        self,
        rag_content: List[Dict],
        instructions: Dict
    ) -> Dict:
        """Process semantic search results."""
        
        if not rag_content:
            return {
                "content_chunks": [],
                "summary": "No relevant content found for your query.",
                "key_concepts": [],
                "source_diversity": {}
            }
        
        # Re-rank and filter content
        ranked_chunks = self._process_and_rank_semantic_content(rag_content, instructions)
        
        # Extract key concepts
        key_concepts = self._extract_key_concepts(ranked_chunks)
        
        # Analyze source diversity
        source_diversity = self._analyze_source_diversity(ranked_chunks)
        
        # Create comprehensive summary
        summary = self._create_semantic_summary(ranked_chunks, key_concepts)
        
        return {
            "content_chunks": ranked_chunks,
            "summary": summary,
            "key_concepts": key_concepts,
            "source_diversity": source_diversity
        }
    
    def _process_and_rank_semantic_content(
        self,
        rag_content: List[Dict],
        instructions: Dict
    ) -> List[Dict]:
        """Re-rank and process semantic content chunks."""
        
        # Filter by quality threshold
        quality_filtered = [
            chunk for chunk in rag_content
            if chunk.get("relevance", {}).get("overall", 0) >= self.min_chunk_quality_score
        ]
        
        # Re-rank based on multiple factors
        reranked_chunks = self._rerank_chunks(quality_filtered, instructions)
        
        # Remove near-duplicates
        deduplicated_chunks = self._remove_near_duplicates(reranked_chunks)
        
        # Limit to max chunks
        final_chunks = deduplicated_chunks[:self.max_chunks]
        
        # Compress content for each chunk
        compressed_chunks = []
        for chunk in final_chunks:
            compressed_chunk = self._compress_chunk_content(chunk, instructions)
            compressed_chunks.append(compressed_chunk)
        
        return compressed_chunks
    
    def _rerank_chunks(self, chunks: List[Dict], instructions: Dict) -> List[Dict]:
        """Re-rank chunks using multiple relevance signals."""
        
        query_lower = instructions.get("query", "").lower()
        query_entities = instructions.get("entities", [])
        intent_signals = instructions.get("intent_signals", {})
        
        for chunk in chunks:
            # Calculate enhanced relevance score
            enhanced_score = self._calculate_enhanced_relevance(
                chunk, query_lower, query_entities, intent_signals
            )
            chunk["enhanced_relevance"] = enhanced_score
        
        # Sort by enhanced relevance
        chunks.sort(key=lambda x: x["enhanced_relevance"], reverse=True)
        
        return chunks
    
    def _calculate_enhanced_relevance(
        self,
        chunk: Dict,
        query_lower: str,
        query_entities: List[Dict],
        intent_signals: Dict
    ) -> float:
        """Calculate enhanced relevance score using multiple factors."""
        
        base_relevance = chunk.get("relevance", {}).get("overall", 0.0)
        content = chunk.get("content", {})
        metadata = chunk.get("metadata", {})
        
        # Factor 1: Content freshness (if applicable)
        freshness_boost = 0.0
        created_at = metadata.get("created_at")
        if created_at and "recent" in intent_signals.get("urgency", ""):
            # Simple freshness boost - could be more sophisticated
            freshness_boost = 0.1
        
        # Factor 2: Content type preference match
        content_type_boost = 0.0
        preferred_type = intent_signals.get("content_preference")
        if preferred_type and content.get("content_type") == preferred_type:
            content_type_boost = 0.15
        
        # Factor 3: Entity alignment
        entity_boost = 0.0
        chunk_text = content.get("text", "").lower()
        for entity in query_entities:
            if entity.get("value", "").lower() in chunk_text:
                entity_boost += 0.05  # Cumulative boost for entity matches
        entity_boost = min(entity_boost, 0.2)  # Cap the boost
        
        # Factor 4: Title prominence
        title_boost = 0.0
        if any(word in content.get("title", "").lower() for word in query_lower.split()):
            title_boost = 0.1
        
        # Factor 5: Content length appropriateness
        length_factor = 1.0
        content_length = metadata.get("chunk_length", 0)
        if content_length < 50:  # Too short
            length_factor = 0.8
        elif content_length > 2000:  # Too long
            length_factor = 0.9
        
        # Combine all factors
        enhanced_score = (
            base_relevance * length_factor + 
            freshness_boost + 
            content_type_boost + 
            entity_boost + 
            title_boost
        )
        
        return min(enhanced_score, 1.0)  # Cap at 1.0
    
    def _remove_near_duplicates(self, chunks: List[Dict]) -> List[Dict]:
        """Remove chunks with very similar content."""
        if len(chunks) <= 1:
            return chunks
        
        unique_chunks = []
        seen_signatures = set()
        
        for chunk in chunks:
            # Create content signature for similarity detection
            content_text = chunk.get("content", {}).get("text", "")
            
            # Simple signature based on first and last 100 characters
            if len(content_text) > 200:
                signature = content_text[:100] + content_text[-100:]
            else:
                signature = content_text
            
            # Normalize signature
            signature = re.sub(r'\s+', ' ', signature.lower().strip())
            signature_hash = hash(signature) % 100000
            
            if signature_hash not in seen_signatures:
                unique_chunks.append(chunk)
                seen_signatures.add(signature_hash)
        
        return unique_chunks
    
    def _compress_chunk_content(self, chunk: Dict, instructions: Dict) -> Dict:
        """Compress individual chunk content while preserving key information."""
        compressed_chunk = chunk.copy()
        content = chunk.get("content", {})
        text = content.get("text", "")
        
        if len(text) > 500:  # Only compress longer chunks
            # Extract key sentences
            compressed_text = self._extract_key_sentences(text, instructions)
            compressed_chunk["content"]["text"] = compressed_text
            compressed_chunk["content"]["original_length"] = len(text)
            compressed_chunk["content"]["compressed"] = True
        
        return compressed_chunk
    
    def _extract_key_sentences(self, text: str, instructions: Dict) -> str:
        """Extract the most relevant sentences from a text chunk."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 3:
            return text
        
        query_words = set(instructions.get("query", "").lower().split())
        
        # Score sentences by query word overlap
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words & sentence_words)
            score = overlap / len(query_words) if query_words else 0
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:3]]
        
        return ". ".join(top_sentences) + "."
    
    def _categorize_structured_content(self, mongodb_content: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize structured content by type."""
        categories = defaultdict(list)
        
        for item in mongodb_content:
            # Determine category based on item properties
            if "course" in str(item.get("title", "")).lower() or item.get("type") == "course":
                categories["courses"].append(item)
            elif "assignment" in str(item.get("title", "")).lower() or item.get("type") == "assignment":
                categories["assignments"].append(item)
            elif "progress" in item or "completion" in item:
                categories["progress"].append(item)
            elif "metric" in item or "count" in item:
                categories["metrics"].append(item)
            else:
                categories["general"].append(item)
        
        return dict(categories)
    
    def _create_structured_section(
        self,
        category: str,
        items: List[Dict],
        instructions: Dict
    ) -> Dict:
        """Create a structured section for a category of items."""
        
        section = {
            "category": category,
            "item_count": len(items),
            "summary": "",
            "key_items": [],
            "details": []
        }
        
        if category == "courses":
            section["summary"] = f"Found {len(items)} course(s) matching your criteria."
            section["key_items"] = [
                {
                    "title": item.get("title", "Untitled"),
                    "category": item.get("category", "Uncategorized"),
                    "id": item.get("id", "")
                }
                for item in items[:5]  # Top 5 courses
            ]
        
        elif category == "metrics":
            section["summary"] = "Course statistics and counts."
            section["key_items"] = [
                {
                    "metric": item.get("metric", ""),
                    "value": item.get("value", 0),
                    "description": item.get("description", "")
                }
                for item in items
            ]
        
        elif category == "progress":
            completed_count = len([item for item in items if item.get("status") == "completed"])
            in_progress_count = len([item for item in items if item.get("status") == "in_progress"])
            section["summary"] = f"Progress summary: {completed_count} completed, {in_progress_count} in progress."
            
        return section
    
    def _create_structured_summary(
        self,
        mongodb_content: List[Dict],
        categorized_content: Dict[str, List[Dict]]
    ) -> str:
        """Create an overall summary of structured content."""
        
        summary_parts = []
        
        for category, items in categorized_content.items():
            if category == "courses" and items:
                summary_parts.append(f"{len(items)} course(s)")
            elif category == "assignments" and items:
                summary_parts.append(f"{len(items)} assignment(s)")
            elif category == "metrics" and items:
                summary_parts.append("statistical data")
        
        if summary_parts:
            return f"Retrieved {', '.join(summary_parts)} based on your query."
        else:
            return "Retrieved structured data matching your query."
    
    def _extract_key_data_points(self, mongodb_content: List[Dict]) -> List[Dict]:
        """Extract key data points from structured content."""
        data_points = []
        
        for item in mongodb_content:
            if item.get("metric"):
                data_points.append({
                    "type": "metric",
                    "key": item.get("metric"),
                    "value": item.get("value"),
                    "context": item.get("description", "")
                })
            elif item.get("title"):
                data_points.append({
                    "type": "item",
                    "key": "title",
                    "value": item.get("title"),
                    "context": item.get("category", "")
                })
        
        return data_points[:10]  # Limit to 10 key data points
    
    def _weighted_merge(
        self,
        structured_summary: Dict,
        semantic_chunks: List[Dict],
        instructions: Dict
    ) -> Dict:
        """Merge content using weighted priority strategy."""
        
        # Determine weights based on query characteristics
        query = instructions.get("query", "").lower()
        
        if any(keyword in query for keyword in ["my", "progress", "count", "how many"]):
            structured_weight = 0.7
            semantic_weight = 0.3
        elif any(keyword in query for keyword in ["explain", "what is", "how does"]):
            structured_weight = 0.3
            semantic_weight = 0.7
        else:
            structured_weight = 0.5
            semantic_weight = 0.5
        
        return {
            "primary_content": structured_summary if structured_weight > semantic_weight else {"content_chunks": semantic_chunks},
            "supporting_content": {"content_chunks": semantic_chunks} if structured_weight > semantic_weight else structured_summary,
            "merge_weights": {
                "structured": structured_weight,
                "semantic": semantic_weight
            }
        }
    
    def _contextual_merge(
        self,
        structured_summary: Dict,
        semantic_chunks: List[Dict],
        instructions: Dict
    ) -> Dict:
        """Merge content with contextual awareness."""
        
        return {
            "structured_data": structured_summary,
            "semantic_content": {"content_chunks": semantic_chunks},
            "merge_strategy": "contextual",
            "context_notes": "Both structured and semantic content provided for comprehensive response."
        }
    
    def _simple_combine(self, structured_summary: Dict, semantic_chunks: List[Dict]) -> Dict:
        """Simple combination of structured and semantic content."""
        
        return {
            "combined_content": {
                "structured": structured_summary,
                "semantic": {"content_chunks": semantic_chunks}
            },
            "merge_strategy": "simple"
        }
    
    def _optimize_for_llm(self, processed_content: Dict, instructions: Dict) -> Dict:
        """Final optimization for LLM context window."""
        
        # Calculate total content length
        total_length = self._calculate_content_length(processed_content)
        
        if total_length <= self.max_context_length:
            return processed_content
        
        # Need to compress further
        optimized_content = self._compress_for_context_limit(processed_content, instructions)
        
        return optimized_content
    
    def _calculate_content_length(self, content: Dict) -> int:
        """Calculate approximate character length of content."""
        
        def count_dict_chars(d):
            total = 0
            for key, value in d.items():
                if isinstance(value, str):
                    total += len(value)
                elif isinstance(value, dict):
                    total += count_dict_chars(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            total += count_dict_chars(item)
                        elif isinstance(item, str):
                            total += len(item)
            return total
        
        return count_dict_chars(content)
    
    def _compress_for_context_limit(self, content: Dict, instructions: Dict) -> Dict:
        """Compress content to fit within context limits."""
        
        # Strategy: Keep the highest quality content and compress the rest
        compressed = content.copy()
        
        # If there are semantic chunks, limit them
        if "content_chunks" in compressed:
            chunks = compressed["content_chunks"]
            # Keep only the top chunks and compress their text
            top_chunks = chunks[:8]  # Reduced from max_chunks
            for chunk in top_chunks:
                text = chunk.get("content", {}).get("text", "")
                if len(text) > 300:
                    # More aggressive compression
                    chunk["content"]["text"] = text[:300] + "..."
            compressed["content_chunks"] = top_chunks
        
        return compressed
    
    def _extract_key_concepts(self, chunks: List[Dict]) -> List[str]:
        """Extract key concepts from semantic content."""
        
        concept_frequency = defaultdict(int)
        
        for chunk in chunks:
            text = chunk.get("content", {}).get("text", "").lower()
            title = chunk.get("content", {}).get("title", "").lower()
            
            # Extract potential concepts (noun phrases, important terms)
            # Simple implementation - could be enhanced with NLP
            words = re.findall(r'\b[a-z]{3,}\b', text + " " + title)
            
            for word in words:
                if len(word) > 3 and word not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'had', 'but', 'not', 'his', 'may', 'use']:
                    concept_frequency[word] += 1
        
        # Return top concepts
        sorted_concepts = sorted(concept_frequency.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, freq in sorted_concepts[:10] if freq > 1]
    
    def _analyze_source_diversity(self, chunks: List[Dict]) -> Dict:
        """Analyze the diversity of sources in the content."""
        
        sources = defaultdict(int)
        categories = defaultdict(int)
        content_types = defaultdict(int)
        
        for chunk in chunks:
            content = chunk.get("content", {})
            metadata = chunk.get("metadata", {})
            
            # Count sources
            resource_id = content.get("resource_id", "unknown")
            sources[resource_id] += 1
            
            # Count categories
            category = content.get("category", "unknown")
            categories[category] += 1
            
            # Count content types
            content_type = content.get("content_type", "unknown")
            content_types[content_type] += 1
        
        return {
            "unique_sources": len(sources),
            "source_distribution": dict(sources),
            "category_distribution": dict(categories),
            "content_type_distribution": dict(content_types),
            "diversity_score": len(sources) / len(chunks) if chunks else 0
        }
    
    def _create_semantic_summary(self, chunks: List[Dict], key_concepts: List[str]) -> str:
        """Create a summary of semantic content."""
        
        if not chunks:
            return "No semantic content available."
        
        source_count = len(set(chunk.get("content", {}).get("resource_id", "") for chunk in chunks))
        concept_list = ", ".join(key_concepts[:5]) if key_concepts else "various topics"
        
        return f"Retrieved {len(chunks)} relevant content pieces from {source_count} sources covering {concept_list}."
    
    def _generate_processing_metadata(
        self,
        rag_results: Dict,
        mongodb_results: Optional[Dict],
        processed_content: Dict,
        optimized_content: Dict
    ) -> Dict:
        """Generate comprehensive processing metadata."""
        
        return {
            "processing_timestamp": datetime.utcnow().isoformat(),
            "input_summary": {
                "rag_results_count": len(rag_results.get("results", [])) if rag_results.get("success") else 0,
                "mongodb_results_count": len(mongodb_results.get("results", [])) if mongodb_results and mongodb_results.get("success") else 0
            },
            "processing_steps": {
                "reranking_applied": True,
                "deduplication_applied": True,
                "compression_applied": self._calculate_content_length(processed_content) > self.max_context_length,
                "optimization_applied": processed_content != optimized_content
            },
            "output_summary": {
                "final_content_length": self._calculate_content_length(optimized_content),
                "content_sections": len(optimized_content.keys()),
                "ready_for_llm": True
            }
        }
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create standardized error response."""
        return {
            "success": False,
            "processed_content": {},
            "content_type": "error",
            "processing_metadata": {"error": error_message},
            "ready_for_synthesis": False
        }
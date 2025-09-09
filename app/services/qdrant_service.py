from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional
from app.models.resource_model import Resource
import uuid


@dataclass
class VectorLoadResult:
    """Result of vector loading operation"""

    success: bool
    points_inserted: int
    total_chunks: int
    errors: List[str]
    message: str


@dataclass
class ResourceLoadResult:
    success: bool
    points: List[Dict[str, Any]]
    errors: List[str]


@dataclass
class VectorResult:
    success: bool
    vector: Optional[List[float]]
    error: Optional[str]


@dataclass
class InsertionResult:
    success_count: int
    errors: List[str]


class VectorDatabaseLoader:
    """Handles loading resources into vector database with improved error handling and performance"""

    def __init__(self, token_credentials, org_id: str):
        self.token_credentials = token_credentials
        self.org_id = org_id
        self.batch_size = 100  # Configurable batch size

    async def load_course_with_content(
        self, course: Resource, sub_items: List[Resource]
    ) -> VectorLoadResult:
        """
        Load course and all its content into vector database with better error handling
        """
        errors = []
        all_points = []

        try:
            # Load main course resource
            course_result = await self._load_single_resource(course)
            if course_result.success:
                all_points.extend(course_result.points)
            else:
                errors.extend(course_result.errors)

            # Load course content (lessons, quizzes, etc.)
            content_result = await self._load_course_content(course, sub_items)
            if content_result.success:
                all_points.extend(content_result.points)
            else:
                errors.extend(content_result.errors)

            # Batch insert all points
            if all_points:
                insertion_result = await self._batch_insert_points(all_points)
                return VectorLoadResult(
                    success=len(errors) == 0,
                    points_inserted=insertion_result.success_count,
                    total_chunks=len(all_points),
                    errors=errors + insertion_result.errors,
                    message=f"Loaded {insertion_result.success_count}/{len(all_points)} chunks",
                )
            else:
                return VectorLoadResult(
                    success=False,
                    points_inserted=0,
                    total_chunks=0,
                    errors=errors + ["No valid content found to vectorize"],
                    message="No content loaded",
                )

        except Exception as e:
            print(f"Error in load_course_with_content: {e}")
            return VectorLoadResult(
                success=False,
                points_inserted=0,
                total_chunks=0,
                errors=[str(e)],
                message="Failed to load course content",
            )

    async def _load_single_resource(self, resource: Resource) -> "ResourceLoadResult":
        """Load a single resource with improved error handling"""
        try:
            resource_data = self._prepare_resource_data(resource)

            # Generate vector
            vector_result = await self._generate_vector_safe(resource_data)
            if not vector_result.success:
                return ResourceLoadResult(
                    success=False,
                    points=[],
                    errors=[f"Vector generation failed: {vector_result.error}"],
                )

            point_payload = {
                "id": str(uuid.uuid4()),
                "payload": resource_data,
                "vector": vector_result.vector,
            }

            return ResourceLoadResult(success=True, points=[point_payload], errors=[])

        except Exception as e:
            print(f"Error loading resource {resource.resource_id}: {e}")
            return ResourceLoadResult(success=False, points=[], errors=[str(e)])

    def _prepare_resource_data(self, resource: Resource) -> Dict[str, Any]:
        """Prepare resource data with chatbot-specific enhancements"""
        resource_data = {
            "resource_id": resource.resource_id,
            "title": resource.title,
            "short_description": resource.short_description,
            "description": resource.description,
            "category": resource.category,
            "type": resource.type,
            "language": resource.language or "en",
            "authors": (
                [author.name for author in resource.authors] if resource.authors else []
            ),
            "publisher_name": resource.publisher.name if resource.publisher else None,
            "created_at": (
                resource.created_at.isoformat() if resource.created_at else None
            ),
            "updated_at": (
                resource.updated_at.isoformat() if resource.updated_at else None
            ),
            "content_type": "main_resource",
            "searchable_content": self._create_searchable_content(resource),
        }

        return {k: v for k, v in resource_data.items() if v is not None}

    async def _generate_vector_safe(
        self, resource_data: Dict[str, Any]
    ) -> "VectorResult":
        """Generate vector with proper error handling and fallbacks"""
        try:
            from app.helpers.vertex_helper import get_query_vector

            # Create text for vectorization
            text_fields = self._extract_text_fields(resource_data)
            combined_text = " ".join(text_fields)

            if not combined_text.strip():
                return VectorResult(
                    success=False,
                    vector=None,
                    error="No text content available for vectorization",
                )

            # Truncate if too long (embedding models have limits)
            if len(combined_text) > 8000:  # Conservative limit
                combined_text = combined_text[:8000]
                print("Text truncated for vectorization")

            vector = get_query_vector(combined_text)

            # Validate vector
            expected_size = 3072
            if len(vector) != expected_size:
                return VectorResult(
                    success=False,
                    vector=None,
                    error=f"Vector size mismatch. Expected {expected_size}, got {len(vector)}",
                )

            return VectorResult(success=True, vector=vector, error=None)

        except Exception as e:
            print(f"Error generating vector: {e}")
            return VectorResult(success=False, vector=None, error=str(e))

    def _create_searchable_content(self, resource: Resource) -> str:
        """Create optimized searchable content for chatbot queries"""
        content_parts = []

        if resource.title:
            content_parts.append(resource.title)
        if resource.short_description:
            content_parts.append(resource.short_description)
        if resource.description and resource.description != resource.short_description:
            content_parts.append(resource.description)

        # Add course-specific keywords for better retrieval
        if resource.type == "course":
            content_parts.extend([
                "course",
                "learning material",
                "educational content",
                "training program"
            ])
            if resource.category:
                content_parts.append(f"category: {resource.category}")

        return " | ".join(content_parts)

    def _extract_text_fields(self, resource_data: Dict[str, Any]) -> List[str]:
        """Extract and prioritize text fields for vectorization"""
        text_fields = []

        # Priority order for text fields
        priority_fields = ["title", "short_description", "description"]
        secondary_fields = ["category", "authors", "publisher_name"]

        # Add high-priority fields first
        for field in priority_fields:
            if resource_data.get(field):
                text_fields.append(str(resource_data[field]))

        # Add secondary fields
        for field in secondary_fields:
            value = resource_data.get(field)
            if value:
                if isinstance(value, list):
                    text_fields.extend([str(v) for v in value])
                else:
                    text_fields.append(str(value))

        return [field for field in text_fields if field.strip()]

    async def _batch_insert_points(
        self, points: List[Dict[str, Any]]
    ) -> "InsertionResult":
        """Insert points in batches for better performance"""
        success_count = 0
        errors = []

        # Process in batches
        for i in range(0, len(points), self.batch_size):
            batch = points[i : i + self.batch_size]
            try:
                await self._insert_batch(batch)
                success_count += len(batch)
                print(f"Successfully inserted batch {i//self.batch_size + 1}")
            except Exception as e:
                error_msg = f"Failed to insert batch {i//self.batch_size + 1}: {str(e)}"
                print(error_msg)
                errors.append(error_msg)

                # Try individual insertions for failed batch
                for point in batch:
                    try:
                        await self._insert_single_point(point)
                        success_count += 1
                    except Exception as point_error:
                        errors.append(
                            f"Failed to insert point {point['id']}: {str(point_error)}"
                        )

        return InsertionResult(success_count=success_count, errors=errors)

    async def _insert_batch(self, batch: List[Dict[str, Any]]):
        """Insert a batch of points"""
        import aiohttp

        target_audience = os.getenv("TARGET_AUDIENCE")
        headers = {
            "Authorization": f"Bearer {self.token_credentials.token}",
            "Content-Type": "application/json",
        }

        url = f"{target_audience}/collections/{self.org_id}/points"
        payload = {"points": batch}

        async with aiohttp.ClientSession() as session:
            async with session.put(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                return await response.json()

    async def _insert_single_point(self, point: Dict[str, Any]):
        """Insert a single point"""
        return await self._insert_batch([point])

    def _prepare_course_metadata(self, course: Resource) -> Dict[str, Any]:
        """Prepare base metadata for course content"""
        return {
            "course_id": course.resource_id,
            "course_title": course.title,
            "course_category": course.category,
            "course_description": course.description,
            "content_type": "course_content",
        }

    async def _load_course_content(
        self, course: Resource, resources: List[Resource]
    ) -> "ResourceLoadResult":
        """Load course content with chunking and better error handling"""
        base_metadata = self._prepare_course_metadata(course)
        all_points = []
        errors = []

        for resource in resources:
            try:
                # Extract and process text chunks
                chunks = self._extract_content_chunks(resource)
                if not chunks:
                    print(f"No chunks found for resource {resource.resource_id}")
                    continue

                # Process chunks with semantic chunking
                processed_chunks = self._process_chunks_semantically(chunks, resource)

                for chunk_data in processed_chunks:
                    chunk_metadata = {**base_metadata, **chunk_data["metadata"]}

                    # Generate vector for chunk
                    vector_result = await self._generate_vector_safe(
                        {
                            "title": chunk_data["title"],
                            "description": chunk_data["metadata"]["original_chunk"],
                        }
                    )

                    if vector_result.success:
                        point_payload = {
                            "id": str(uuid.uuid4()),
                            "payload": chunk_metadata,
                            "vector": vector_result.vector,
                        }
                        all_points.append(point_payload)
                    else:
                        errors.append(
                            f"Vector generation failed for chunk in {resource.resource_id}"
                        )

            except Exception as e:
                print(f"Error processing resource {resource.resource_id}: {e}")
                errors.append(f"Failed to process {resource.resource_id}: {str(e)}")
                continue

        return ResourceLoadResult(
            success=len(errors) == 0, points=all_points, errors=errors
        )

    def _extract_content_chunks(self, resource: Resource) -> List[str]:
        """
        Enhanced content extraction with better structure handling
        """
        text_chunks = []

        if resource.type == "lesson" and resource.document:
            chunks = self._extract_lesson_content(resource.document)
            text_chunks.extend(chunks)
        elif resource.type == "quiz":
            chunks = self._extract_quiz_content(resource)
            text_chunks.extend(chunks)
        elif resource.type in ["section", "assignment"]:
            chunks = self._extract_structured_content(resource)
            text_chunks.extend(chunks)

        # Add resource description as a chunk if substantial
        if resource.description and len(resource.description.strip()) > 50:
            text_chunks.append(resource.description.strip())

        # Filter out empty or very short chunks
        return [chunk for chunk in text_chunks if chunk and len(chunk.strip()) > 10]

    def _extract_lesson_content(self, document: Dict[str, Any]) -> List[str]:
        """Extract content from lesson documents with better structure handling"""
        chunks = []

        blocks = document.get("items", []) if isinstance(document, dict) else []

        for block in blocks:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")

            if block_type == "text":
                text_content = self._extract_text_block_content(block)
                chunks.extend(text_content)
            elif block_type in ["quote", "list"]:
                content = self._extract_special_block_content(block)
                if content:
                    chunks.append(content)

        return chunks

    def _extract_text_block_content(self, block: Dict[str, Any]) -> List[str]:
        """Extract content from text blocks"""
        chunks = []
        items = block.get("items", [])

        for item in items:
            if not isinstance(item, dict):
                continue

            # Handle heading
            heading = item.get("heading", "").strip()
            if heading:
                chunks.append(heading)

            # Handle content blocks
            content_blocks = item.get("content", {}).get("blocks", [])
            if content_blocks:
                for cb in content_blocks:
                    if isinstance(cb, dict) and cb.get("type") == "paragraph":
                        text = cb.get("data", {}).get("text", "").strip()
                        if text:
                            chunks.append(text)
            else:
                paragraph = item.get("paragraph", "").strip()
                if paragraph:
                    chunks.append(paragraph)

        return chunks

    def _extract_special_block_content(self, block: Dict[str, Any]) -> Optional[str]:
        """Extract content from special blocks like quotes, lists"""
        # Implement based on your specific block structure
        # This is a placeholder - customize based on your data structure
        return None

    def _extract_quiz_content(self, resource: Resource) -> List[str]:
        """Extract quiz content for better search context"""
        chunks = []

        if resource.description:
            chunks.append(f"Quiz: {resource.description}")

        # Add more quiz-specific extraction logic here
        return chunks

    def _extract_structured_content(self, resource: Resource) -> List[str]:
        """Extract content from structured resources like sections, assignments"""
        chunk_parts = []

        if resource.title:
            chunk_parts.append(f"Title: {resource.title}")
        if resource.description:
            chunk_parts.append(f"Description: {resource.description}")

        return [" | ".join(chunk_parts)] if chunk_parts else []

    def _process_chunks_semantically(
        self,
        chunks: List[str],
        resource: Resource,
    ) -> List[Dict[str, Any]]:
        """
        Process chunks with semantic considerations for better chatbot responses
        """
        processed_chunks = []

        for i, chunk_text in enumerate(chunks):
            if not chunk_text or len(chunk_text.strip()) < 10:  # Skip very short chunks
                continue

            # Add context for better retrieval
            chunk_with_context = self._add_context_to_chunk(chunk_text, resource, i)

            processed_chunks.append(
                {
                    "text": chunk_with_context,
                    "title": resource.title,
                    "metadata": {
                        "resource_id": resource.resource_id,
                        "title": resource.title,
                        "type": resource.type,
                        "chunk_index": i,
                        "chunk_text": chunk_with_context,
                        "original_chunk": chunk_text,
                        "chunk_length": len(chunk_text),
                        "language": resource.language or "en",
                    },
                }
            )

        return processed_chunks

    def _add_context_to_chunk(
        self, chunk_text: str, resource: Resource, chunk_index: int
    ) -> str:
        """Add contextual information to chunks for better retrieval"""
        context_parts = []

        # Add resource context
        if resource.title:
            context_parts.append(f"From: {resource.title}")

        if resource.type:
            context_parts.append(f"Type: {resource.type}")

        if resource.category:
            context_parts.append(f"Category: {resource.category}")

        context = " | ".join(context_parts)

        # Format with context
        return f"{context}\n\n{chunk_text}" if context else chunk_text
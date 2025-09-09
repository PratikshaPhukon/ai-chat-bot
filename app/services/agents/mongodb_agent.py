"""
MongoDB Agent - Step 3a of Multi-Agent Pipeline
Handles structured database queries for operational data retrieval.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from app.helpers.mongodb_helper import mongodb_helper
from app.models.resource_model import Resource
from app.models.user_model import User

logger = logging.getLogger(__name__)


class MongoDBAgent:
    """
    Executes structured database queries based on routing instructions.
    Handles user-specific data, course information, progress tracking, etc.
    """
    
    def __init__(self):
        self.max_results = 100
        self.default_timeout = 30  # seconds
        
    async def execute_query(
        self, 
        instructions: Dict, 
        user_context: Dict
    ) -> Dict:
        """
        Execute MongoDB query based on processing instructions.
        
        Args:
            instructions: Processing instructions from IntentRouterAgent
            user_context: User information (org_id, user_id, etc.)
            
        Returns:
            Query results with metadata
        """
        try:
            org_id = user_context.get("org_id")
            user_id = user_context.get("user_id")
            
            if not org_id:
                return self._create_error_response("Missing organization context")
            
            # Ensure database connection
            await mongodb_helper._ensure_tenant_initialized(org_id)
            
            # Parse query requirements
            query_type = self._determine_query_type(instructions)
            filters = instructions.get("database_filters", {})
            
            # Execute appropriate query
            if query_type == "user_courses":
                results = await self._query_user_courses(org_id, user_id, filters)
            elif query_type == "course_details":
                results = await self._query_course_details(org_id, filters)
            elif query_type == "user_progress":
                results = await self._query_user_progress(org_id, user_id, filters)
            elif query_type == "assignments":
                results = await self._query_assignments(org_id, user_id, filters)
            elif query_type == "course_count":
                results = await self._query_course_count(org_id, filters)
            elif query_type == "course_list":
                results = await self._query_course_list(org_id, filters)
            else:
                # Fallback to general query
                results = await self._execute_general_query(org_id, user_id, instructions)
            
            return {
                "success": True,
                "results": results,
                "query_type": query_type,
                "result_count": len(results) if isinstance(results, list) else 1,
                "execution_time": datetime.utcnow().isoformat(),
                "filters_applied": filters
            }
            
        except Exception as e:
            logger.error(f"MongoDB query execution failed: {e}")
            return self._create_error_response(f"Database query failed: {str(e)}")
    
    def _determine_query_type(self, instructions: Dict) -> str:
        """Determine the type of query to execute based on instructions."""
        query = instructions.get("query", "").lower()
        entities = instructions.get("entities", [])
        filters = instructions.get("database_filters", {})
        
        # Check for user-specific course queries
        if any(keyword in query for keyword in ["my courses", "enrolled", "taking"]):
            return "user_courses"
        
        # Check for course details queries
        elif any(keyword in query for keyword in ["course details", "about course", "course info"]):
            return "course_details"
        
        # Check for progress queries
        elif any(keyword in query for keyword in ["progress", "completion", "finished"]):
            return "user_progress"
        
        # Check for assignment queries
        elif any(keyword in query for keyword in ["assignment", "homework", "due", "deadline"]):
            return "assignments"
        
        # Check for count queries
        elif any(keyword in query for keyword in ["how many", "count", "number of"]):
            return "course_count"
        
        # Check for list queries
        elif any(keyword in query for keyword in ["list", "show all", "all courses"]):
            return "course_list"
        
        # Default fallback
        else:
            return "general_query"
    
    async def _query_user_courses(
        self, 
        org_id: str, 
        user_id: str, 
        filters: Dict
    ) -> List[Dict]:
        """Query courses specific to a user."""
        try:
            # Get user information first
            user = await User.find_one(User.id == user_id) if user_id else None
            
            if not user:
                return []
            
            # Build query for user's courses
            query_filters = {"type": "course"}
            
            # Apply status filters if specified
            if filters.get("status"):
                status = filters["status"]
                if status in ["completed", "in_progress", "not_started"]:
                    # This would require cross-referencing with user progress data
                    # Simplified implementation for now
                    query_filters["status"] = {"$in": [1, 3]}  # Active courses
            
            # Apply category filters
            if filters.get("category_hint"):
                query_filters["category"] = {
                    "$regex": filters["category_hint"], 
                    "$options": "i"
                }
            
            courses = await Resource.find(query_filters).limit(self.max_results).to_list()
            
            # Transform to standardized format
            return [self._format_course_result(course) for course in courses]
            
        except Exception as e:
            logger.error(f"Error querying user courses: {e}")
            return []
    
    async def _query_course_details(self, org_id: str, filters: Dict) -> List[Dict]:
        """Query detailed information about specific courses."""
        try:
            query_filters = {"type": "course"}
            
            # Apply category filters
            if filters.get("category_hint"):
                query_filters["$or"] = [
                    {"category": {"$regex": filters["category_hint"], "$options": "i"}},
                    {"title": {"$regex": filters["category_hint"], "$options": "i"}},
                    {"description": {"$regex": filters["category_hint"], "$options": "i"}}
                ]
            
            courses = await Resource.find(query_filters).limit(20).to_list()
            
            results = []
            for course in courses:
                # Get sub-items for each course
                sub_items = await Resource.find({
                    "document.course_id": course.resource_id
                }).to_list()
                
                course_data = self._format_course_result(course)
                course_data["sub_items_count"] = len(sub_items)
                course_data["has_content"] = len(sub_items) > 0
                results.append(course_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying course details: {e}")
            return []
    
    async def _query_user_progress(
        self, 
        org_id: str, 
        user_id: str, 
        filters: Dict
    ) -> List[Dict]:
        """Query user progress information."""
        try:
            # This is a simplified implementation
            # In a real system, you'd have separate progress tracking collections
            
            # Get user data
            user = await User.find_one(User.id == user_id) if user_id else None
            if not user:
                return []
            
            # Get courses and simulate progress data
            courses = await Resource.find({"type": "course"}).limit(20).to_list()
            
            progress_data = []
            for course in courses:
                # Simulate progress calculation
                progress_item = {
                    "course_id": course.resource_id,
                    "course_title": course.title,
                    "progress_percentage": 0,  # Would calculate from actual data
                    "status": "not_started",   # Would get from user progress records
                    "last_accessed": None,
                    "completion_date": None
                }
                progress_data.append(progress_item)
            
            return progress_data
            
        except Exception as e:
            logger.error(f"Error querying user progress: {e}")
            return []
    
    async def _query_assignments(
        self, 
        org_id: str, 
        user_id: str, 
        filters: Dict
    ) -> List[Dict]:
        """Query assignments and their due dates."""
        try:
            query_filters = {"type": "assignment"}
            
            # Apply due date filters
            due_date_filter = filters.get("due_date")
            if due_date_filter == "today":
                today = datetime.utcnow().date()
                # Note: This would need proper date field handling in real implementation
            elif due_date_filter == "this_week":
                # Similar date range filtering
                pass
            elif due_date_filter == "overdue":
                # Filter for overdue assignments
                pass
            
            assignments = await Resource.find(query_filters).limit(50).to_list()
            
            return [self._format_assignment_result(assignment) for assignment in assignments]
            
        except Exception as e:
            logger.error(f"Error querying assignments: {e}")
            return []
    
    async def _query_course_count(self, org_id: str, filters: Dict) -> List[Dict]:
        """Get count of courses based on filters."""
        try:
            query_filters = {"type": "course"}
            
            # Apply category filters
            if filters.get("category_hint"):
                query_filters["category"] = {
                    "$regex": filters["category_hint"], 
                    "$options": "i"
                }
            
            count = await Resource.find(query_filters).count()
            
            return [{
                "metric": "course_count",
                "value": count,
                "filters_applied": filters,
                "description": f"Total courses matching criteria: {count}"
            }]
            
        except Exception as e:
            logger.error(f"Error getting course count: {e}")
            return [{"metric": "course_count", "value": 0, "error": str(e)}]
    
    async def _query_course_list(self, org_id: str, filters: Dict) -> List[Dict]:
        """Get list of all courses."""
        try:
            query_filters = {"type": "course"}
            
            # Apply filters
            if filters.get("status"):
                query_filters["status"] = {"$in": [1, 3]}  # Active statuses
            
            if filters.get("category_hint"):
                query_filters["$or"] = [
                    {"category": {"$regex": filters["category_hint"], "$options": "i"}},
                    {"title": {"$regex": filters["category_hint"], "$options": "i"}}
                ]
            
            courses = await Resource.find(query_filters).limit(self.max_results).to_list()
            
            return [self._format_course_summary(course) for course in courses]
            
        except Exception as e:
            logger.error(f"Error querying course list: {e}")
            return []
    
    async def _execute_general_query(
        self, 
        org_id: str, 
        user_id: str, 
        instructions: Dict
    ) -> List[Dict]:
        """Execute general query when specific type isn't determined."""
        try:
            # Default to course search
            courses = await Resource.find({"type": "course"}).limit(10).to_list()
            return [self._format_course_result(course) for course in courses]
            
        except Exception as e:
            logger.error(f"Error executing general query: {e}")
            return []
    
    def _format_course_result(self, course: Resource) -> Dict:
        """Format course data for consistent output."""
        return {
            "id": course.resource_id,
            "title": course.title or "Untitled Course",
            "description": course.short_description or course.description or "",
            "category": course.category or "Uncategorized",
            "type": course.type,
            "language": course.language or "en",
            "created_at": course.created_at.isoformat() if course.created_at else None,
            "updated_at": course.updated_at.isoformat() if course.updated_at else None,
            "authors": [author.name for author in course.authors] if course.authors else [],
            "tags": course.tags or [],
            "access_type": course.access_type,
            "status": course.status
        }
    
    def _format_course_summary(self, course: Resource) -> Dict:
        """Format course data for summary lists."""
        return {
            "id": course.resource_id,
            "title": course.title or "Untitled Course",
            "category": course.category or "Uncategorized",
            "description": (course.short_description or course.description or "")[:200] + "..." if (course.short_description or course.description or "") else "",
            "type": course.type
        }
    
    def _format_assignment_result(self, assignment: Resource) -> Dict:
        """Format assignment data for output."""
        return {
            "id": assignment.resource_id,
            "title": assignment.title or "Untitled Assignment",
            "description": assignment.description or "",
            "type": assignment.type,
            "due_date": None,  # Would extract from actual due date field
            "status": "pending",  # Would get from user progress
            "course_id": assignment.document.get("course_id") if assignment.document else None
        }
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create standardized error response."""
        return {
            "success": False,
            "results": [],
            "query_type": "error",
            "result_count": 0,
            "error": error_message,
            "execution_time": datetime.utcnow().isoformat()
        }
    
    async def get_user_context(self, user_id: str, org_id: str) -> Optional[Dict]:
        """Get additional user context for query personalization."""
        try:
            user = await User.find_one(User.id == user_id) if user_id else None
            if not user:
                return None
                
            return {
                "user_id": str(user.id),
                "role": user.role,
                "preferences": {
                    "language": user.language_preference,
                    "interests": user.user_interest or []
                },
                "progress_data": user.resources or []
            }
            
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return None


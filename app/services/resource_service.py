from typing import List, Optional
from fastapi import HTTPException
from app.helpers.mongodb_helper import mongodb_helper
from app.models.resource_model import Resource


async def get_resource_by_id(resource_id: str, org_id: str) -> Optional[Resource]:
    """Get resource details by resource_id from tenant-specific MongoDB."""
    try:
        await mongodb_helper._ensure_tenant_initialized(org_id)
        resource = await Resource.find_one(Resource.resource_id == resource_id)
        return resource
    except Exception as e:
        print(f"Error fetching resource from MongoDB for tenant {org_id}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching resource data")


async def get_sub_items_by_course_id(course_id: str, org_id: str) -> List[Resource]:
    """Get all sub-items of a course by course_id from tenant-specific MongoDB."""
    try:
        await mongodb_helper._ensure_tenant_initialized(org_id)
        sub_items = await Resource.find({"document.course_id": course_id}).to_list()

        return sub_items

    except Exception as e:
        print(f"Error fetching sub-items for course {course_id} in org {org_id}: {e}")
        return []


async def get_all_courses(org_id: str) -> List[Resource]:
    """Get all courses from tenant-specific MongoDB."""
    try:
        await mongodb_helper._ensure_tenant_initialized(org_id)
        # Find resources that are courses (main resources, not sub-items)
        courses = await Resource.find(
            Resource.type == "course"
        ).to_list()
        
        return courses

    except Exception as e:
        print(f"Error fetching courses for org {org_id}: {e}")
        return []
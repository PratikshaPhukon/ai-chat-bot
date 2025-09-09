from fastapi import Request, HTTPException
from app.helpers.qdrant_helper import check_and_create_collection
from app.helpers.vertex_helper import get_credentials, initialize_vertex_ai
from app.middleware.auth_middleware import get_user_info_from_request
from app.services.qdrant_service import VectorDatabaseLoader
from app.services.resource_service import (
    get_resource_by_id,
    get_sub_items_by_course_id,
)


async def get_resource_by_id_controller(request: Request, resource_id: str) -> dict:
    """Get resource details controller."""
    try:
        _id, org_id = get_user_info_from_request(request)
        print(f"Processing request for org_id: {org_id}, resource_id: {resource_id}")

        # Get resource details from service
        resource = await get_resource_by_id(resource_id, org_id)
        sub_items = await get_sub_items_by_course_id(resource_id, org_id)

        if not resource:
            return {
                "code": 404,
                "status": "error",
                "message": "Resource not found",
                "data": None,
            }

        credentials, token_credentials = get_credentials()

        check_and_create_collection(org_id, token_credentials)
        initialize_vertex_ai(credentials)

        # Load resource into Qdrant
        loader = VectorDatabaseLoader(token_credentials, org_id)
        result = await loader.load_course_with_content(resource, sub_items)

        if result.success:
            print(f"Resource loaded successfully: {result.message}")
        else:
            print(f"Resource loading failed: {result.errors}")

        return {
            "code": 200,
            "status": "success",
            "message": "Resource details retrieved successfully.",
            "data": {"resource": resource.dict(), "sub_items": sub_items},
        }

    except HTTPException as e:
        return {
            "code": e.status_code,
            "status": "error",
            "message": e.detail,
            "data": None,
        }
    except Exception as e:
        return {
            "code": 500,
            "status": "error",
            "message": f"Internal server error: {str(e)}",
            "data": None,
        }
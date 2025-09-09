from fastapi import Request, HTTPException
from app.middleware.auth_middleware import get_user_info_from_request
from app.services.user_service import (
    get_user_by_id,
)

async def get_user_details_controller(request: Request) -> dict:
    """Get user details controller."""
    try:
        _id, org_id = get_user_info_from_request(request)

        # Get user details from service
        user = await get_user_by_id(_id, org_id)
        if not user:
            return {
                "code": 404,
                "status": "error",
                "message": "User not found",
                "data": None,
            }

        return {
            "code": 200,
            "status": "success",
            "message": "User details retrieved successfully.",
            "data": user.dict(),
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
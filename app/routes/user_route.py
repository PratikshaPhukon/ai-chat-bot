from fastapi import APIRouter, Request, Query
from pydantic import BaseModel
from typing import Optional
from app.controllers.user_controller import (
    get_user_details_controller,
)
from app.middleware.auth_middleware import (
    require_auth,
)

router = APIRouter()


class CreateUserRequest(BaseModel):
    email: str
    name: Optional[str] = None
    role: str = "learner"
    is_active: bool = True


class UpdateUserRequest(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


@router.get("/details", summary="Get user details from MongoDB")
@require_auth
async def get_user_details(request: Request):
    """Get user details from MongoDB using email from request state."""
    return await get_user_details_controller(request)
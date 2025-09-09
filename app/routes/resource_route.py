from fastapi import APIRouter, Request, Path
from app.controllers.resource_controller import (
    get_resource_by_id_controller,
)
from app.middleware.auth_middleware import (
    require_auth,
)

router = APIRouter()


@router.post("/load-resource/{resource_id}", summary="Load Resource by resource ID")
@require_auth
async def get_resource_by_id(
    request: Request,
    resource_id: str = Path(..., description="The resource ID to retrieve"),
):
    """Get resource details from MongoDB using resource_id."""
    return await get_resource_by_id_controller(request, resource_id)
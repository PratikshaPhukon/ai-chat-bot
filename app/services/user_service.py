from typing import Optional
from fastapi import HTTPException, Request
from beanie import PydanticObjectId
from app.helpers.mongodb_helper import mongodb_helper
from app.models.user_model import User


async def get_user_by_id(_id: str, org_id: str) -> Optional[User]:
    """Get user details by id from tenant-specific MongoDB."""
    try:
        await mongodb_helper._ensure_tenant_initialized(org_id)
        user = await User.find_one(User.id == PydanticObjectId(_id))
        return user
    except Exception as e:
        print(f"Error fetching user from MongoDB for tenant {org_id}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching user data")
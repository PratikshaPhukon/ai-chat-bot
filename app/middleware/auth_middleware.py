import os
import jwt
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Optional
from functools import wraps
from starlette.middleware.base import BaseHTTPMiddleware
import inspect

class JWTAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.jwt_secret = os.getenv("JWT_SECRET")
        if not self.jwt_secret:
            raise ValueError("JWT_SECRET environment variable is required")

    def verify_token(self, token: str) -> Optional[dict]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )

    async def dispatch(self, request: Request, call_next):
        """Middleware function to verify JWT token and extract tenant information"""
        # Skip authentication for certain paths if needed
        if request.url.path in ["/docs", "/openapi.json", "/health"]:
            return await call_next(request)

        # Get token from header
        auth_token = request.headers.get("x-auth-token")

        if not auth_token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "x-auth-token header is required"},
            )

        try:
            # Verify token
            payload = self.verify_token(auth_token)

            request.state.user = payload

            return await call_next(request)

        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"},
            )


def require_auth(func):
    """Decorator to wrap both sync and async route handlers"""

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper


def get_user_info_from_request(request: Request) -> dict:
    """Helper function to extract user info from request state."""
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User information not found in request",
        )
    _id = user.get("_id")
    org_id = user.get("org_id")
    if user.get("channel") != "b2b":
        org_id = os.getenv("MONGO_DB_MASTER_DB")
    return _id, org_id
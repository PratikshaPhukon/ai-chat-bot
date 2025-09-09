from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from beanie import Document
from bson import ObjectId
from pydantic import BaseModel, Field, EmailStr, field_validator

# 🔹 Sub-models
class PhoneInfo(BaseModel):
    phone_number: Optional[str]
    country_code: str = "+91"
    valid: bool = False


class EmailInfo(BaseModel):
    id: EmailStr
    verified: bool = False
    valid: bool = True


class PasswordInfo(BaseModel):
    key: Optional[str]
    salt: Optional[str]


class OnboardingInfo(BaseModel):
    is_show: bool = False
    current_step: int = 0
    startedAt: Optional[datetime] = None
    completedAt: Optional[datetime] = None


class MembershipInfo(BaseModel):
    type: str = Field(default="free", enum=["free", "premium"])
    validity_date: datetime = Field(
        default_factory=lambda: datetime.utcnow() - timedelta(days=1)
    )
    channel: str = Field(default="b2c", enum=["b2b", "b2c"])


class AssignedResourceItem(BaseModel):
    resource_id: str
    due_date: Optional[str] = None


class ResourceItem(BaseModel):
    resource_id: str
    validity_date: Optional[str] = None
    isFavorite: Optional[bool] = None
    isWishlist: Optional[bool] = None
    status: Optional[str] = Field(default=None, enum=["inProgress", "completed"])
    metaData: Optional[Dict[str, Any]] = None
    updatedAt: datetime = Field(default_factory=datetime.utcnow)
    type: str = Field(
        default="book",
        enum=[
            "book",
            "video",
            "collection",
            "course",
            "document",
            "learning-path",
            "micro-learning",
        ],
    )


class OTPSecret(BaseModel):
    otp: Optional[str]
    expiry: Optional[datetime]


class LocationInfo(BaseModel):
    city: Optional[str]
    country: Optional[str]


# 🔹 Main User model
class User(Document):
    id: Optional[str] = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    first_name: Optional[str]
    last_name: Optional[str]
    phone: Optional[PhoneInfo] = None
    email: EmailInfo
    password: Optional[PasswordInfo]
    org_id: Optional[str]
    purchase: Optional[Any] = None
    status: str = Field(default="active", enum=["active", "suspended"])
    role: str = Field(
        default="Learner", enum=["Learner", "Admin", "Author", "Group Admin"]
    )
    department: Optional[str] = None
    job_title: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    assigned_resources: Optional[List[AssignedResourceItem]] = Field(
        default_factory=list
    )
    profile_image: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login_at: datetime = Field(default_factory=datetime.utcnow)
    last_active_at: datetime = Field(default_factory=datetime.utcnow)
    login_type: str = Field(
        default="N", enum=["N", "G", "S"]
    )  # N-normal, G-google, S-SSO
    firebase_token: str = ""
    user_type: str = Field(
        default="student_ug",
        enum=[
            "student_ug",
            "student_pg",
            "educator",
            "publisher",
            "professional",
            "learner",
            "ops_admin",
            "ops_blog_writer",
            "ops_growth",
        ],
    )
    onboarding: Optional[OnboardingInfo] = Field(default_factory=OnboardingInfo)
    products: List[str] = Field(default_factory=lambda: ["lms"])
    membership: MembershipInfo = Field(default_factory=MembershipInfo)
    country: Optional[str] = None
    language_preference: Optional[str] = None
    user_interest: List[str] = Field(default_factory=list)
    groups: List[str] = Field(default_factory=list)
    resources: List[ResourceItem] = Field(default_factory=list)
    otp_secret: Optional[OTPSecret] = None
    achievements: Optional[Dict[str, Any]] = None
    manager: Optional[Any] = None
    location: Optional[LocationInfo] = None
    hierarchy_path: List[str] = Field(default_factory=list)
    level: int = 0  # 0-user, 1-manager, 2-admin
    attributes: List[Any] = Field(default_factory=list)

    @field_validator("id", mode="before")   
    @classmethod
    def convert_objectid_to_str(cls, v):
        if isinstance(v, ObjectId):
            return str(v)
        return v
    class Settings:
        name = "users"
        indexes = ["email.id", "org_id", "role", "user_type"]

    class Config:
        json_schema_extra = {
            "example": {
                "email": {"id": "user@example.com", "verified": False, "valid": True},
                "role": "Learner",
                "membership": {
                    "type": "free",
                    "validity_date": "2024-01-01T00:00:00Z",
                    "channel": "b2c",
                },
            }
        }
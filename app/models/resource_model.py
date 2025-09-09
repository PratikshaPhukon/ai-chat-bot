from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from beanie import Document
from pydantic import BaseModel, Field


# 🔹 Sub-models
class PublisherInfo(BaseModel):
    url: Optional[str]
    name: Optional[str]


class AuthorInfo(BaseModel):
    id: Optional[str]
    name: Optional[str]
    description: Optional[str] = None


class TemplateInfo(BaseModel):
    is_template: bool = False


class SeoMeta(BaseModel):
    title: Optional[str]
    description: Optional[str]
    keywords: Optional[str]


# 🔹 Main Resource model
class Resource(Document):
    url: Optional[str] = Field(unique=True)
    resource_id: Optional[str]
    title: Optional[str]
    subtitle: Optional[str] = None
    short_description: Optional[str] = None
    description: Optional[str] = None
    published_date: Optional[str] = None
    publisher: Optional[PublisherInfo] = None
    creator: Optional[str] = None
    authors: List[AuthorInfo] = Field(default_factory=list)
    view_count: int = 0
    category: Optional[str]
    subcategory: Optional[str]
    tags: List[str] = Field(default_factory=list)
    cover_image: Optional[str] = None
    visibility: int = 5
    copyright: Optional[str] = None
    can_rent: bool = False

    access_specifiers: Literal["private", "public", "protected"] = "private"
    type: Literal[
        "book",
        "collection",
        "video",
        "course",
        "lesson",
        "section",
        "quiz",
        "scorm12",
        "pdf",
        "learning-path",
        "document",
        "micro-learning",
        "question-bank",
        "assignment",
    ] = "book"

    access_type: Literal["free", "paid"] = "free"
    channel: Literal["B2C", "B2B", "ALL"] = "ALL"

    document: Optional[Dict[str, Any]]
    template: Optional[TemplateInfo] = Field(default_factory=TemplateInfo)
    status: int = Field(
        default=1,
        description="1-active, 2-disabled, 3-published, 4-submitted, 5-only meta, 6-Offline, 12-moderation",
    )

    mrp: Dict[str, Any] = Field(default_factory=dict)
    seo_meta: Optional[SeoMeta] = None
    price: Dict[str, Any] = Field(default_factory=dict)
    groups: List[str] = Field(default_factory=list)

    whitelist_countries: List[Literal["ALL", "IN"]] = Field(
        default_factory=lambda: ["ALL"]
    )
    blacklist_countries: List[Literal["IN"]] = Field(default_factory=list)

    language: Literal["en", "hi"] = "en"
    education_levels: List[
        Literal[
            "pre_school",
            "lower_primary",
            "upper_primary",
            "middle_school",
            "high_school",
            "ug_pg",
            "career_tech",
        ]
    ] = Field(default_factory=lambda: ["ug_pg"])

    is_submitted: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "resources"
        indexes = ["resource_id", "type", "category", "tags"]

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Sample Book",
                "type": "book",
                "language": "en",
                "access_type": "free",
                "channel": "B2C",
                "tags": ["science", "technology"],
            }
        }
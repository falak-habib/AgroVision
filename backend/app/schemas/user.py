"""
Pydantic v2 schemas for User requests and responses.
"""

from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class UserCreate(BaseModel):
    """Request schema for new user registration."""

    full_name: str = Field(
        ...,
        min_length=2,
        max_length=255,
        description="User's full name",
        examples=["Aarav Sharma"],
    )
    email: EmailStr = Field(
        ...,
        description="Valid email address",
        examples=["farmer@example.com"],
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Account password (minimum 8 characters)",
        examples=["StrongPass123!"],
    )

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 2:
            raise ValueError("Full name must contain at least 2 non-whitespace characters.")
        return stripped


class UserResponse(BaseModel):
    """Safe user profile response schema (never exposes hashed_password)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    email: EmailStr
    is_active: bool
    created_at: datetime
    updated_at: datetime

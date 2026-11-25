"""
Pydantic v2 schemas for JWT authentication tokens.
"""

from typing import Optional
from pydantic import BaseModel, Field


class Token(BaseModel):
    """OAuth2 bearer token response schema."""

    access_token: str = Field(..., description="Encoded JWT access token")
    token_type: str = Field(default="bearer", description="Token type (bearer)")


class TokenData(BaseModel):
    """Decoded JWT payload claims."""

    sub: Optional[str] = Field(default=None, description="Token subject (User ID)")

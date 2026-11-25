"""
Security utilities for password hashing (Argon2) and JWT token management (PyJWT).
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union
import jwt
from pwdlib import PasswordHash

from backend.app.core.config import get_settings

# Recommended Argon2 password hash context
password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """Hash a plaintext password securely using Argon2."""
    return password_hash.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against an Argon2 hash."""
    if not plain_password or not hashed_password:
        return False
    return password_hash.verify(plain_password, hashed_password)


def create_access_token(
    subject: Union[str, int],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT access token.
    Includes 'sub' (subject user ID), 'iat' (issued at), and 'exp' (expiration).
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    if expires_delta is not None:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    payload: Dict[str, Any] = {
        "sub": str(subject),
        "iat": now,
        "exp": expire,
    }
    encoded_jwt = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT access token.
    Verifies signature, expiration, and presence of required claims ('exp', 'sub').
    Raises jwt.PyJWTError subclasses on malformed, expired, or invalid tokens.
    """
    settings = get_settings()
    payload = jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
        options={"require": ["exp", "sub"]},
    )
    return payload

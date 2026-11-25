"""
Authentication and user management service layer.
Encapsulates database queries, credential verification, and token issuance.
"""

from typing import Optional
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.core.security import create_access_token, hash_password, verify_password
from backend.app.models.user import User
from backend.app.schemas.user import UserCreate


class AuthServiceError(Exception):
    """Base exception for authentication service errors."""
    pass


class EmailAlreadyExistsError(AuthServiceError):
    """Raised when attempting to register with an email that is already taken."""
    pass


class InvalidCredentialsError(AuthServiceError):
    """Raised when email does not exist or password does not match."""
    pass


class InactiveAccountError(AuthServiceError):
    """Raised when an inactive user attempts to authenticate."""
    pass


def normalize_email(email: str) -> str:
    """Normalize email address consistently (strip whitespace, lowercase)."""
    return email.strip().lower()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Retrieve a user by their normalized email address."""
    normalized = normalize_email(email)
    stmt = select(User).where(User.email == normalized)
    return db.execute(stmt).scalar_one_or_none()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Retrieve a user by primary key ID."""
    stmt = select(User).where(User.id == user_id)
    return db.execute(stmt).scalar_one_or_none()


def create_user(db: Session, user_in: UserCreate) -> User:
    """
    Create a new user account with an Argon2-hashed password.
    Prevents duplicate email registrations.
    """
    normalized_email = normalize_email(str(user_in.email))

    if get_user_by_email(db, normalized_email) is not None:
        raise EmailAlreadyExistsError("An account with this email already exists.")

    user = User(
        full_name=user_in.full_name.strip(),
        email=normalized_email,
        hashed_password=hash_password(user_in.password),
        is_active=True,
    )

    try:
        db.add(user)
        db.commit()
        db.refresh(user)
    except IntegrityError as exc:
        db.rollback()
        raise EmailAlreadyExistsError("An account with this email already exists.") from exc

    return user


def authenticate_user(db: Session, email: str, password: str) -> User:
    """
    Authenticate user credentials.
    Raises InvalidCredentialsError on unknown email or wrong password.
    Raises InactiveAccountError if the account is deactivated.
    """
    user = get_user_by_email(db, email)
    if user is None or not verify_password(password, user.hashed_password):
        raise InvalidCredentialsError("Incorrect email or password.")

    if not user.is_active:
        raise InactiveAccountError("Inactive user account.")

    return user


def issue_access_token(user: User) -> str:
    """Generate a JWT access token for an authenticated user."""
    return create_access_token(subject=user.id)

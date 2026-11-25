"""
Authentication routes for user registration, login, and profile retrieval.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user
from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.schemas.auth import Token
from backend.app.schemas.user import UserCreate, UserResponse
from backend.app.services.auth_service import (
    EmailAlreadyExistsError,
    InactiveAccountError,
    InvalidCredentialsError,
    authenticate_user,
    create_user,
    issue_access_token,
)

router = APIRouter()


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with full name, email, and password.",
)
def register(
    user_in: UserCreate,
    db: Session = Depends(get_db),
) -> UserResponse:
    """Register a new AgroVision user."""
    try:
        user = create_user(db, user_in)
    except EmailAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return UserResponse.model_validate(user)


@router.post(
    "/login",
    response_model=Token,
    status_code=status.HTTP_200_OK,
    summary="User Login",
    description="Authenticate with email (as OAuth2 username) and password to obtain a JWT Bearer token.",
)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> Token:
    """Authenticate user credentials and return a signed JWT access token."""
    try:
        user = authenticate_user(
            db,
            email=form_data.username,
            password=form_data.password,
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except InactiveAccountError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    access_token = issue_access_token(user)
    return Token(access_token=access_token, token_type="bearer")


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Current Authenticated User",
    description="Retrieve the safe profile of the currently authenticated user.",
)
def read_current_user(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """Return the authenticated user's profile."""
    return UserResponse.model_validate(current_user)

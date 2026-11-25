"""
Automated tests for AgroVision JWT Authentication & User Management.
Uses an isolated in-memory SQLite database via FastAPI dependency overrides
so development/production PostgreSQL data is never modified.
"""

from datetime import timedelta
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.core.security import decode_access_token, create_access_token, verify_password
from backend.app.db.base import Base
from backend.app.db.session import get_db
from backend.app.main import app
from backend.app.models.user import User


@pytest.fixture()
def db_session():
    """Create a fresh isolated in-memory SQLite database and session per test."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    Base.metadata.create_all(bind=test_engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture()
def client(db_session: Session):
    """Provide a FastAPI TestClient wired to the isolated test database session."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# -----------------------------------------------------------------------------
# REGISTRATION TESTS
# -----------------------------------------------------------------------------

def test_register_success(client: TestClient, db_session: Session):
    """Verify successful registration, email normalization, Argon2 hashing, and safe response."""
    payload = {
        "full_name": "  Aarav Sharma  ",
        "email": "Aarav.Sharma@Example.com",
        "password": "SecurePassword123!",
    }
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert data["full_name"] == "Aarav Sharma"
    assert data["email"] == "aarav.sharma@example.com"
    assert data["is_active"] is True
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data
    assert "hashed_password" not in data
    assert "password" not in data

    # Verify password is hashed in database
    db_user = db_session.query(User).filter(User.email == "aarav.sharma@example.com").first()
    assert db_user is not None
    assert db_user.hashed_password != payload["password"]
    assert db_user.hashed_password.startswith("$argon2")
    assert verify_password(payload["password"], db_user.hashed_password) is True


def test_register_duplicate_email_rejected(client: TestClient):
    """Verify duplicate email (including case-insensitive variants) is rejected with 409."""
    payload = {
        "full_name": "Farmer One",
        "email": "farmer@agrovision.org",
        "password": "Password123!",
    }
    res1 = client.post("/api/v1/auth/register", json=payload)
    assert res1.status_code == 201

    # Duplicate with different casing
    res2 = client.post(
        "/api/v1/auth/register",
        json={
            "full_name": "Farmer Two",
            "email": "FARMER@AgroVision.ORG",
            "password": "AnotherPassword123!",
        },
    )
    assert res2.status_code == 409
    assert "already exists" in res2.json()["detail"].lower()


@pytest.mark.parametrize("bad_email", ["not-an-email", "missing@domain", "@no-user.com", ""])
def test_register_invalid_email_rejected(client: TestClient, bad_email: str):
    """Verify malformed email addresses are rejected with 422."""
    response = client.post(
        "/api/v1/auth/register",
        json={"full_name": "Valid Name", "email": bad_email, "password": "Password123!"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("bad_password", ["short", "1234567", ""])
def test_register_invalid_password_rejected(client: TestClient, bad_password: str):
    """Verify passwords shorter than 8 characters are rejected with 422."""
    response = client.post(
        "/api/v1/auth/register",
        json={"full_name": "Valid Name", "email": "valid@example.com", "password": bad_password},
    )
    assert response.status_code == 422


def test_register_invalid_full_name_rejected(client: TestClient):
    """Verify whitespace-only or too-short full_name is rejected with 422."""
    response = client.post(
        "/api/v1/auth/register",
        json={"full_name": "   ", "email": "valid@example.com", "password": "Password123!"},
    )
    assert response.status_code == 422


# -----------------------------------------------------------------------------
# LOGIN TESTS
# -----------------------------------------------------------------------------

def test_login_success_and_token_claims(client: TestClient):
    """Verify successful OAuth2 login returns a valid JWT with sub, exp, and iat claims."""
    client.post(
        "/api/v1/auth/register",
        json={
            "full_name": "Priya Patel",
            "email": "priya@example.com",
            "password": "MySecretPassword99!",
        },
    )

    login_res = client.post(
        "/api/v1/auth/login",
        data={"username": "PRIYA@example.com", "password": "MySecretPassword99!"},
    )
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert token_data["token_type"] == "bearer"
    assert "access_token" in token_data

    # Verify JWT claims
    decoded = decode_access_token(token_data["access_token"])
    assert "sub" in decoded
    assert "exp" in decoded
    assert "iat" in decoded


def test_login_incorrect_password_rejected(client: TestClient):
    """Verify wrong password returns HTTP 401 with WWW-Authenticate header and generic error."""
    client.post(
        "/api/v1/auth/register",
        json={"full_name": "User", "email": "user@example.com", "password": "CorrectPassword123!"},
    )
    res = client.post(
        "/api/v1/auth/login",
        data={"username": "user@example.com", "password": "WrongPassword123!"},
    )
    assert res.status_code == 401
    assert res.headers.get("www-authenticate") == "Bearer"
    assert res.json()["detail"] == "Incorrect email or password."


def test_login_unknown_email_rejected(client: TestClient):
    """Verify unknown email returns HTTP 401 with generic error identical to wrong password."""
    res = client.post(
        "/api/v1/auth/login",
        data={"username": "nobody@example.com", "password": "SomePassword123!"},
    )
    assert res.status_code == 401
    assert res.headers.get("www-authenticate") == "Bearer"
    assert res.json()["detail"] == "Incorrect email or password."


def test_login_inactive_account_rejected(client: TestClient, db_session: Session):
    """Verify deactivated user account cannot log in (HTTP 403)."""
    client.post(
        "/api/v1/auth/register",
        json={"full_name": "Inactive User", "email": "inactive@example.com", "password": "Password123!"},
    )
    user = db_session.query(User).filter(User.email == "inactive@example.com").first()
    user.is_active = False
    db_session.commit()

    res = client.post(
        "/api/v1/auth/login",
        data={"username": "inactive@example.com", "password": "Password123!"},
    )
    assert res.status_code == 403
    assert "inactive" in res.json()["detail"].lower()


# -----------------------------------------------------------------------------
# PROTECTED ROUTE (/auth/me) TESTS
# -----------------------------------------------------------------------------

def test_me_with_valid_token(client: TestClient):
    """Verify valid Bearer token retrieves authenticated user's profile without password hash."""
    client.post(
        "/api/v1/auth/register",
        json={"full_name": "Me User", "email": "me@example.com", "password": "Password123!"},
    )
    token = client.post(
        "/api/v1/auth/login",
        data={"username": "me@example.com", "password": "Password123!"},
    ).json()["access_token"]

    res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    data = res.json()
    assert data["email"] == "me@example.com"
    assert data["full_name"] == "Me User"
    assert "hashed_password" not in data


def test_me_missing_token_rejected(client: TestClient):
    """Verify missing Authorization header returns HTTP 401 with WWW-Authenticate."""
    res = client.get("/api/v1/auth/me")
    assert res.status_code == 401
    assert res.headers.get("www-authenticate") == "Bearer"


def test_me_invalid_token_rejected(client: TestClient):
    """Verify malformed or tampered token returns HTTP 401."""
    res = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer invalid.jwt.token"})
    assert res.status_code == 401
    assert res.headers.get("www-authenticate") == "Bearer"


def test_me_expired_token_rejected(client: TestClient):
    """Verify expired JWT returns HTTP 401."""
    reg = client.post(
        "/api/v1/auth/register",
        json={"full_name": "Expired User", "email": "exp@example.com", "password": "Password123!"},
    )
    user_id = reg.json()["id"]
    expired_token = create_access_token(subject=user_id, expires_delta=timedelta(seconds=-10))

    res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
    assert res.status_code == 401
    assert res.headers.get("www-authenticate") == "Bearer"


def test_me_inactive_user_rejected(client: TestClient, db_session: Session):
    """Verify user deactivated after token issuance cannot access protected route."""
    client.post(
        "/api/v1/auth/register",
        json={"full_name": "Deactivated", "email": "deact@example.com", "password": "Password123!"},
    )
    token = client.post(
        "/api/v1/auth/login",
        data={"username": "deact@example.com", "password": "Password123!"},
    ).json()["access_token"]

    # Deactivate user in DB
    user = db_session.query(User).filter(User.email == "deact@example.com").first()
    user.is_active = False
    db_session.commit()

    res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 403


def test_me_user_isolation(client: TestClient):
    """Verify users can only retrieve their own profile using their respective tokens."""
    client.post(
        "/api/v1/auth/register",
        json={"full_name": "User A", "email": "a@example.com", "password": "Password123!"},
    )
    client.post(
        "/api/v1/auth/register",
        json={"full_name": "User B", "email": "b@example.com", "password": "Password123!"},
    )

    token_a = client.post(
        "/api/v1/auth/login", data={"username": "a@example.com", "password": "Password123!"}
    ).json()["access_token"]
    token_b = client.post(
        "/api/v1/auth/login", data={"username": "b@example.com", "password": "Password123!"}
    ).json()["access_token"]

    res_a = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token_a}"}).json()
    res_b = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token_b}"}).json()

    assert res_a["email"] == "a@example.com"
    assert res_b["email"] == "b@example.com"
    assert res_a["id"] != res_b["id"]


# -----------------------------------------------------------------------------
# DATABASE & MIGRATION TESTS
# -----------------------------------------------------------------------------

def test_users_table_schema_and_email_uniqueness(db_session: Session):
    """Verify users table columns, indexes, and DB-level unique email constraint."""
    inspector = inspect(db_session.get_bind())
    assert "users" in inspector.get_table_names()

    columns = {col["name"] for col in inspector.get_columns("users")}
    assert {"id", "full_name", "email", "hashed_password", "is_active", "created_at", "updated_at"}.issubset(columns)

    # DB-level uniqueness check
    u1 = User(full_name="One", email="dup@example.com", hashed_password="h1")
    u2 = User(full_name="Two", email="dup@example.com", hashed_password="h2")
    db_session.add(u1)
    db_session.commit()

    db_session.add(u2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

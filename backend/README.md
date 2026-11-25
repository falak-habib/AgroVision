# AgroVision Backend Service

The backend service for **AgroVision**, an AI-powered crop disease screening and agricultural assistance system. Built on **FastAPI** with modular architecture, strict typing, centralized configuration, asynchronous routing, and a robust **PostgreSQL / SQLAlchemy 2.0 / Alembic** database foundation.

---

## 1. Quickstart & Local Development

### 1.1 Virtual Environment Activation

#### Windows (PowerShell):
```powershell
.\backend\.venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt):
```cmd
backend\.venv\Scripts\activate.bat
```

#### Linux / macOS:
```bash
source backend/.venv/bin/activate
```

---

### 1.2 Dependency Installation

Ensure all required dependencies are installed within the active virtual environment:
```bash
pip install -r backend/requirements.txt
```

---

### 1.3 Local Environment Configuration

Copy the example environment configuration to `.env` (optional for local dev as sensible defaults are embedded):
```bash
# Windows PowerShell:
Copy-Item .env.example .env

# Linux/macOS:
cp .env.example .env
```

> **Note:** Never commit `.env` or production secrets to Git.

---

### 1.4 Starting PostgreSQL with Docker Compose

A lightweight development PostgreSQL instance can be started using Docker Compose from the repository root:

```bash
# Start PostgreSQL in background
docker compose up -d

# Check PostgreSQL container logs
docker compose logs -f postgres

# Stop PostgreSQL container
docker compose down
```

The database container runs **PostgreSQL 16** with a persistent local volume (`postgres_data`) on port `5432`. Default development connection URL:
```
postgresql+psycopg://postgres:postgres@localhost:5432/agrovision
```

---

### 1.5 Starting the Development Server

From the repository root, start the Uvicorn ASGI server with hot reloading enabled:
```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

---

## 2. Database Architecture & Alembic Migrations

### 2.1 Connection Architecture & Session Management
- **SQLAlchemy 2.0 & Psycopg 3:** Configured via `backend/app/db/session.py` using `postgresql+psycopg://...`.
- **Connection Pooling:** Pre-configured with `pool_pre_ping=True`, `pool_size=10`, `max_overflow=10`, and `pool_timeout=30` to prevent stale connection drops.
- **Dependency Injection:** Database sessions are requested via `db: Session = Depends(get_db)`. Each session is automatically committed or rolled back and safely closed in a `try...finally` block.
- **Shared Declarative Base:** `backend/app/db/base.py` exports `Base = DeclarativeBase` for all future domain models. Table creation via `Base.metadata.create_all()` is strictly avoided in favor of Alembic migrations.

### 2.2 Alembic Migration Commands
Run all Alembic commands from the repository root:

```bash
# Create an auto-generated migration revision
alembic revision --autogenerate -m "migration description"

# Apply all pending migrations to the database
alembic upgrade head

# Roll back the most recent migration
alembic downgrade -1

# View migration history
alembic history

# View current database revision head
alembic current
```

> **Security Note:** `alembic.ini` contains no hardcoded database credentials. `backend/alembic/env.py` dynamically injects the configured `DATABASE_URL` from the application's centralized Pydantic Settings.

---

## 3. Authentication & User Management (JWT)

### 3.1 Authentication Architecture
- **Password Hashing:** Uses **Argon2id** via `pwdlib[argon2]`. Plaintext passwords are never stored or returned.
- **Stateless Tokens:** Uses **PyJWT** (`HS256` by default) to issue signed Bearer tokens containing `sub` (user ID), `iat` (issued at), and `exp` (expiration).
- **Protected Routes:** Depend on `get_current_user` (`backend/app/api/deps.py`), which validates the Bearer token, loads the user from PostgreSQL, and verifies `is_active=True`.

### 3.2 JWT Environment Variables
Add these variables to `.env` (see `.env.example` for placeholders):
- `JWT_SECRET_KEY`: Cryptographically random 32+ byte secret key (never commit real secrets).
- `JWT_ALGORITHM`: Signing algorithm (default: `HS256`).
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token lifetime in minutes (default: `30`).

### 3.3 Authentication Endpoints

#### Register a New User — `POST /api/v1/auth/register` (HTTP 201)
Request (`application/json`):
```json
{
  "full_name": "Aarav Sharma",
  "email": "farmer@example.com",
  "password": "StrongPassword123!"
}
```
Response (`201 Created`):
```json
{
  "id": 1,
  "full_name": "Aarav Sharma",
  "email": "farmer@example.com",
  "is_active": true,
  "created_at": "2026-11-25T10:50:00Z",
  "updated_at": "2026-11-25T10:50:00Z"
}
```

#### Login — `POST /api/v1/auth/login` (HTTP 200)
Request (`application/x-www-form-urlencoded`, OAuth2 Password Flow):
- `username`: `farmer@example.com`
- `password`: `StrongPassword123!`

Response (`200 OK`):
```json
{
  "access_token": "<jwt_token>",
  "token_type": "bearer"
}
```

#### Get Current User Profile — `GET /api/v1/auth/me` (HTTP 200)
Header: `Authorization: Bearer <jwt_token>`
Returns the authenticated user's safe profile (`UserResponse`).

### 3.4 Using Bearer Tokens in Swagger UI
1. Open `http://127.0.0.1:8000/docs`.
2. Register a user via `POST /api/v1/auth/register`.
3. Click the **Authorize** button at the top-right of Swagger UI.
4. Enter your registered **email** in the `username` field and your **password** in the `password` field, then click **Authorize**.
5. Swagger will automatically call `/api/v1/auth/login` and attach the `Authorization: Bearer <token>` header to protected requests such as `GET /api/v1/auth/me`.

---

## 4. API Endpoints & Health Check System

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| **`/api/v1/health`** | `GET` | **Liveness probe.** Returns HTTP 200 indicating the backend service is operational. **Completely decoupled from database status.** |
| **`/api/v1/health/ready`** | `GET` | **Readiness probe.** Verifies centralized configuration and executes `SELECT 1` on PostgreSQL. Returns **HTTP 200** if ready, or **HTTP 503** if database is unavailable. Never exposes credentials. |
| **`/docs`** | `GET` | Interactive **Swagger UI** API documentation. |
| **`/redoc`** | `GET` | Alternative **ReDoc** API documentation. |
| **`/api/v1/openapi.json`** | `GET` | Raw OpenAPI 3.1 specification schema. |

### Sample Readiness Response (Database Connected — HTTP 200):
```json
{
  "status": "ready",
  "app_name": "AgroVision API",
  "version": "0.1.0",
  "environment": "development",
  "checks": {
    "configuration": "ok",
    "database": "ok"
  }
}
```

### Sample Readiness Response (Database Disconnected — HTTP 503):
```json
{
  "status": "not_ready",
  "app_name": "AgroVision API",
  "version": "0.1.0",
  "environment": "development",
  "checks": {
    "configuration": "ok",
    "database": "unavailable"
  }
}
```

---

## 5. Running Automated Tests

### 5.1 Backend Unit & Integration Tests (Health, DB, Alembic, Auth)
Runs the full backend test suite:
```bash
pytest backend/tests -v
```
Run authentication tests specifically:
```bash
pytest backend/tests/test_auth.py -v
```

> **Note:** The unit test suite utilizes in-memory SQLite and mock connection interfaces. An externally running PostgreSQL instance is **not** required to execute the test suite.

### 4.2 ML Integration Tests
Rerun the ML integration suite to ensure zero regressions across all 31 model screening tests:
```bash
python -m backend.ml.tests.run_all_tests
```

---

## 5. Directory Structure

```
backend/
├── __init__.py
├── requirements.txt             # Pinned backend dependencies
├── README.md                    # Backend service documentation
├── alembic/                     # Alembic migration environment
│   ├── env.py                   # Dynamic configuration & Base.metadata hook
│   ├── script.py.mako           # Migration template
│   └── versions/                # Generated revision files
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app factory, CORS, and lifecycle management
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py            # Centralized pydantic-settings with DATABASE_URL
│   ├── db/
│   │   ├── __init__.py
│   │   ├── base.py              # Shared SQLAlchemy 2.0 DeclarativeBase
│   │   └── session.py           # Engine, connection pooling, and get_db dependency
│   └── api/
│       ├── __init__.py
│       └── routes/
│           ├── __init__.py      # Central API router
│           └── health.py        # /api/v1/health & /api/v1/health/ready probes
├── ml/                          # Pre-validated ML inference and validation layer
└── tests/
    ├── __init__.py
    ├── test_alembic.py          # Alembic configuration & metadata tests
    ├── test_database.py         # SQLAlchemy engine, session, & connectivity tests
    └── test_health.py           # Liveness, readiness, and metadata tests
```

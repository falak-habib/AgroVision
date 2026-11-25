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

## 3. API Endpoints & Health Check System

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

## 4. Running Automated Tests

### 4.1 Backend Unit & Integration Tests
Runs the full backend test suite covering configuration, database session lifecycle, connection error handling, Alembic setup, and health/readiness endpoints:
```bash
pytest backend/tests -v
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

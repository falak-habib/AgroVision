# AgroVision — Phase 2.3: JWT Authentication & User Management Report

**Phase:** 2.3 — JWT Authentication & User Management  
**Branch:** `apoorva` (Verified active, `main` untouched)  
**Date:** November 25, 2026  

---

## 1. Implementation Summary

Phase 2.3 implements secure, stateless JWT-based authentication and user account management for the AgroVision FastAPI backend:
- Created a SQLAlchemy 2.0 `User` model (`users` table) using the shared `DeclarativeBase`.
- Implemented Argon2 password hashing (`pwdlib[argon2]`) and HS256 JWT token management (`PyJWT[crypto]`).
- Built Pydantic v2 validation schemas (`UserCreate`, `UserResponse`, `Token`, `TokenData`).
- Separated business logic into an authentication service layer (`backend/app/services/auth_service.py`) with email normalization, duplicate prevention, and generic error handling.
- Implemented `get_current_user` OAuth2 Bearer dependency (`backend/app/api/deps.py`).
- Exposed `/api/v1/auth/register`, `/api/v1/auth/login`, and `/api/v1/auth/me` via the centralized API router.
- Generated and applied Alembic migration `ca5845949abc_create_users_table.py` to PostgreSQL.
- Verified with 42 passing backend tests (21 new auth tests + 21 existing tests) and 31 passing ML integration tests.

---

## 2. Files Created

1. [`backend/app/models/__init__.py`](file:///c:/Users/KIIT/Desktop/AgroVision/backend/app/models/__init__.py) — Models package exports.
2. [`backend/app/models/user.py`](file:///c:/Users/KIIT/Desktop/AgroVision/backend/app/models/user.py) — SQLAlchemy 2.0 `User` model (`users` table).
3. [`backend/app/core/security.py`](file:///c:/Users/KIIT/Desktop/AgroVision/backend/app/core/security.py) — Argon2 password hashing/verification and PyJWT encode/decode utilities.
4. [`backend/app/schemas/__init__.py`](file:///c:/Users/KIIT/Desktop/AgroVision/backend/app/schemas/__init__.py) — Schemas package exports.
5. [`backend/app/schemas/user.py`](file:///c:/Users/KIIT/Desktop/AgroVision/backend/app/schemas/user.py) — `UserCreate` and `UserResponse` Pydantic v2 schemas.
6. [`backend/app/schemas/auth.py`](file:///c:/Users/KIIT/Desktop/AgroVision/backend/app/schemas/auth.py) — `Token` and `TokenData` schemas.
7. [`backend/app/services/__init__.py`](file:///c:/Users/KIIT/Desktop/AgroVision/backend/app/services/__init__.py) — Services package exports.
8. [`backend/app/services/auth_service.py`](file:///c:/Users/KIIT/Desktop/AgroVision/backend/app/services/auth_service.py) — User creation, lookup, credential authentication, and token issuance.
9. [`backend/app/api/deps.py`](file:///c:/Users/KIIT/Desktop/AgroVision/backend/app/api/deps.py) — `oauth2_scheme` and `get_current_user` dependency.
10. [`backend/app/api/routes/auth.py`](file:///c:/Users/KIIT/Desktop/AgroVision/backend/app/api/routes/auth.py) — `/register`, `/login`, and `/me` route handlers.
11. [`backend/alembic/versions/ca5845949abc_create_users_table.py`](file:///c:/Users/KIIT/Desktop/AgroVision/backend/alembic/versions/ca5845949abc_create_users_table.py) — Alembic migration creating the `users` table.
12. [`backend/tests/test_auth.py`](file:///c:/Users/KIIT/Desktop/AgroVision/backend/tests/test_auth.py) — 21 automated tests covering registration, login, JWT validation, user isolation, and DB schema.
13. [`AUTH_IMPLEMENTATION_REPORT.md`](file:///c:/Users/KIIT/Desktop/AgroVision/AUTH_IMPLEMENTATION_REPORT.md) — This implementation report.

---

## 3. Files Modified

1. [`backend/app/core/config.py`](file:///c:/Users/KIIT/Desktop/AgroVision/backend/app/core/config.py) — Added `JWT_SECRET_KEY`, `JWT_ALGORITHM`, and `ACCESS_TOKEN_EXPIRE_MINUTES`.
2. [`.env.example`](file:///c:/Users/KIIT/Desktop/AgroVision/.env.example) — Added documented JWT placeholder environment variables.
3. [`backend/app/api/routes/__init__.py`](file:///c:/Users/KIIT/Desktop/AgroVision/backend/app/api/routes/__init__.py) — Registered `auth.router` under `/auth`.
4. [`backend/alembic/env.py`](file:///c:/Users/KIIT/Desktop/AgroVision/backend/alembic/env.py) — Imported `backend.app.models` so `Base.metadata` discovers `User`.
5. [`backend/tests/test_alembic.py`](file:///c:/Users/KIIT/Desktop/AgroVision/backend/tests/test_alembic.py) — Verified `users` table presence in `target_metadata.tables`.
6. [`backend/README.md`](file:///c:/Users/KIIT/Desktop/AgroVision/backend/README.md) — Added authentication architecture, endpoint examples, Swagger Bearer auth guide, and test commands.

*Unmodified:* `ml/`, `ml/artifacts/`, `backend/ml/`, and `backend/app/api/routes/health.py` remain intact.

---

## 4. Database Schema (`users` Table)

| Column | Type | Constraints / Indexes |
| :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Indexed (`ix_users_id`) |
| `full_name` | `VARCHAR(255)` | `NOT NULL` |
| `email` | `VARCHAR(255)` | `NOT NULL`, Unique Index (`ix_users_email`) |
| `hashed_password` | `VARCHAR(255)` | `NOT NULL` (Argon2id hash) |
| `is_active` | `BOOLEAN` | `NOT NULL`, server default `'true'` |
| `created_at` | `TIMESTAMPTZ` | `NOT NULL`, server default `now()` |
| `updated_at` | `TIMESTAMPTZ` | `NOT NULL`, server default `now()`, `onupdate=now()` |

---

## 5. Authentication Architecture

```
Client Request (Bearer <JWT>)
       │
       ▼
[OAuth2PasswordBearer] ──► Extracts token from Authorization header
       │
       ▼
[decode_access_token()] ─► Validates signature, exp, and sub via PyJWT
       │
       ▼
[get_user_by_id(db, id)] ► Queries PostgreSQL via shared get_db() session
       │
       ▼
[is_active Check] ───────► Rejects deactivated users with HTTP 403
       │
       ▼
Authenticated User Model ──► Injected into protected route handler
```

---

## 6. JWT Configuration

Defined in `backend/app/core/config.py`:
- `JWT_SECRET_KEY`: Loaded from environment / `.env`. Defaults to `secrets.token_urlsafe(32)` in single-process dev when unset (no hardcoded secret).
- `JWT_ALGORITHM`: `"HS256"`
- `ACCESS_TOKEN_EXPIRE_MINUTES`: `30`
- Claims included: `sub` (stringified user ID), `iat` (UTC timestamp), `exp` (UTC expiration).

---

## 7. API Endpoint Documentation

| Method | Path | Auth Required | Status Codes | Description |
| :---: | :--- | :---: | :---: | :--- |
| `POST` | `/api/v1/auth/register` | No | `201`, `409`, `422` | Registers user, hashes password with Argon2, returns `UserResponse`. |
| `POST` | `/api/v1/auth/login` | No (OAuth2 Form) | `200`, `401`, `403` | Authenticates email/password, returns Bearer `Token`. |
| `GET` | `/api/v1/auth/me` | Yes (Bearer JWT) | `200`, `401`, `403` | Returns authenticated user's `UserResponse`. |

---

## 8. Migration Details

- **Revision ID:** `ca5845949abc`
- **File:** `backend/alembic/versions/ca5845949abc_create_users_table.py`
- **Command Executed:** `alembic revision --autogenerate -m "create users table"` followed by `alembic upgrade head`.
- **Current Head:** `ca5845949abc (head)` verified against local PostgreSQL.

---

## 9. Test Results

### 9.1 Backend Test Suite (`pytest backend/tests -v`)
- **Total Collected:** 42 tests
- **Passed:** **42**
- **Failed:** **0**
- **Breakdown:**
  - `test_alembic.py`: 4 passed
  - `test_auth.py`: 21 passed (registration, duplicate/invalid inputs, Argon2 hashing, login, claims, inactive accounts, missing/invalid/expired tokens, user isolation, DB schema/uniqueness)
  - `test_database.py`: 8 passed
  - `test_health.py`: 9 passed

### 9.2 ML Integration Test Suite (`python -m backend.ml.tests.run_all_tests`)
- **Total Run:** 31 tests
- **Passed:** **31**
- **Failed:** **0**
- **Errors:** **0**

### 9.3 Live Application Verification
- `/api/v1/health` -> `200 healthy`
- `/api/v1/health/ready` -> `200 ready {'configuration': 'ok', 'database': 'ok'}`
- `/docs` -> `200`
- OpenAPI paths include `/api/v1/auth/register`, `/api/v1/auth/login`, `/api/v1/auth/me`.

---

## 10. Security Considerations

1. Passwords are hashed using Argon2id (`pwdlib[argon2]`) and never logged or returned in any schema (`UserResponse` excludes `hashed_password`).
2. Login returns generic `"Incorrect email or password."` for both unknown emails and wrong passwords to prevent account enumeration.
3. Emails are normalized (`.strip().lower()`) before lookup and persistence.
4. `JWT_SECRET_KEY` is never hardcoded; production deployments must supply a strong random secret via environment variables.
5. Token decoding enforces signature, `exp`, and `sub` claims.

---

## 11. Known Limitations

1. **Single-Process Ephemeral Secret Fallback:** If `JWT_SECRET_KEY` is not set in `.env`, each Uvicorn worker generates its own ephemeral secret. Multi-worker deployments must set `JWT_SECRET_KEY` explicitly in `.env`.
2. **No Refresh Tokens / Token Revocation:** Access tokens are stateless and valid until `exp` (30 minutes). Refresh tokens and token blacklisting are deferred to future phases if needed.
3. **No Role-Based Access Control (RBAC) Yet:** All users currently share standard active-user status (`is_active`).

---

## 12. Manual Verification Instructions

1. Start the server from repository root:
   ```powershell
   .\backend\.venv\Scripts\uvicorn.exe backend.app.main:app --reload --host 127.0.0.1 --port 8000
   ```
2. Open Swagger UI at `http://127.0.0.1:8000/docs`.
3. **Register:** Expand `POST /api/v1/auth/register`, click **Try it out**, submit `full_name`, `email`, and `password` ($\ge 8$ chars). Confirm HTTP `201`.
4. **Login:** Click **Authorize** (top-right), enter your email in `username` and password in `password`, and click **Authorize** (or test `POST /api/v1/auth/login` directly).
5. **Protected Route:** Expand `GET /api/v1/auth/me`, click **Try it out** $\rightarrow$ **Execute**. Confirm HTTP `200` returning your user profile without `hashed_password`.

---

## 13. Git Branch & Working-Tree Status

- Active branch: `apoorva` (`main` untouched).
- Changes are uncommitted and unpushed per Phase 2.3 rules.

"""
Database session management and connection pooling for AgroVision.
"""

import logging
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import get_settings

logger = logging.getLogger("agrovision.db")

settings = get_settings()

# Connection pooling configuration
engine_kwargs = {"pool_pre_ping": True}
if settings.DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs.update({
        "pool_size": 10,
        "max_overflow": 10,
        "pool_timeout": 30,
    })

engine = create_engine(settings.DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a SQLAlchemy database session
    and guarantees it is closed after the request finishes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connectivity(db_session: Session = None) -> bool:
    """
    Perform a lightweight 'SELECT 1' query to verify database connectivity.
    Does not expose sensitive credentials or internal error details.
    """
    if db_session is not None:
        try:
            db_session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.warning("Database connectivity check failed: %s", type(e).__name__)
            return False

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning("Database engine connectivity check failed: %s", type(e).__name__)
        return False

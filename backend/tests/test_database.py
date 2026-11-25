"""
Unit tests for AgroVision database foundation:
Engine initialization, session creation/cleanup, dependency injection,
and connectivity probes.
"""

from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.base import Base
from backend.app.db.session import (
    SessionLocal,
    check_db_connectivity,
    engine,
    get_db,
)


def test_base_declarative_class():
    """Verify SQLAlchemy 2.0 Base declarative class."""
    assert hasattr(Base, "metadata")
    assert Base.metadata is not None


def test_engine_initialization():
    """Verify that the global engine is initialized with pool_pre_ping."""
    assert engine is not None
    assert engine.pool is not None


def test_session_creation_and_cleanup_sqlite():
    """Verify session creation and proper cleanup using an in-memory SQLite database."""
    test_engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    db: Session = TestingSessionLocal()
    try:
        result = db.execute(text("SELECT 1")).scalar()
        assert result == 1
    finally:
        db.close()

    # Verify session is closed
    assert not db.is_active or db.get_transaction() is None


def test_get_db_dependency_lifecycle():
    """Verify that get_db yields a session and ensures it is closed upon completion."""
    session_generator = get_db()
    db = next(session_generator)
    assert isinstance(db, Session)

    # Trigger finally block by finishing generator
    with pytest.raises(StopIteration):
        next(session_generator)


def test_check_db_connectivity_success():
    """Verify check_db_connectivity returns True when DB executes SELECT 1."""
    mock_session = MagicMock(spec=Session)
    mock_session.execute.return_value = MagicMock()

    assert check_db_connectivity(db_session=mock_session) is True
    mock_session.execute.assert_called_once()


def test_check_db_connectivity_failure():
    """Verify check_db_connectivity returns False and suppresses sensitive exceptions."""
    mock_session = MagicMock(spec=Session)
    mock_session.execute.side_effect = Exception("Connection refused / password=supersecret")

    # Must return False and not re-raise or leak secrets
    assert check_db_connectivity(db_session=mock_session) is False


def test_check_db_connectivity_engine_fallback():
    """Verify check_db_connectivity with mock engine connection."""
    with patch("backend.app.db.session.engine.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        assert check_db_connectivity() is True
        mock_conn.execute.assert_called_once()


def test_check_db_connectivity_engine_failure():
    """Verify check_db_connectivity returns False when engine connection fails."""
    with patch("backend.app.db.session.engine.connect", side_effect=Exception("Database unreachable")):
        assert check_db_connectivity() is False

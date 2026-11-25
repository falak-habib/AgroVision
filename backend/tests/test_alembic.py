"""
Tests for Alembic configuration and metadata integration.
"""

from pathlib import Path
from alembic.config import Config

from backend.app.core.config import get_settings
from backend.app.db.base import Base


def test_alembic_ini_configuration():
    """Verify alembic.ini points to backend/alembic and has no hardcoded credentials."""
    ini_path = Path("alembic.ini")
    assert ini_path.exists(), "alembic.ini must exist at repository root."

    content = ini_path.read_text(encoding="utf-8")
    assert "sqlalchemy.url" not in content or content.count("sqlalchemy.url =") == 0, (
        "alembic.ini must not contain hardcoded database credentials."
    )
    assert "script_location = backend/alembic" in content

    alembic_cfg = Config("alembic.ini")
    assert alembic_cfg.get_main_option("script_location") == "backend/alembic"


def test_alembic_target_metadata():
    """Verify Alembic target metadata references Base.metadata and includes users table."""
    from backend.alembic.env import target_metadata

    assert target_metadata is Base.metadata
    assert "users" in target_metadata.tables


def test_alembic_script_mako_template():
    """Verify migration template exists and contains upgrade and downgrade callables."""
    mako_path = Path("backend/alembic/script.py.mako")
    assert mako_path.exists(), "Alembic script.py.mako must exist."

    content = mako_path.read_text(encoding="utf-8")
    assert "def upgrade()" in content
    assert "def downgrade()" in content


def test_database_url_dynamic_injection():
    """Verify settings.DATABASE_URL is used by Alembic config."""
    settings = get_settings()
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

    assert alembic_cfg.get_main_option("sqlalchemy.url") == settings.DATABASE_URL

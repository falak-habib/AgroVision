"""
Declarative base for SQLAlchemy 2.0 models in AgroVision.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Shared DeclarativeBase for all AgroVision database models.
    Future domain models (Users, Predictions, etc.) will inherit from this base class.
    """
    pass

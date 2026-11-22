"""
Custom exception hierarchy for AgroVision ML inference.
"""

class AgroVisionError(Exception):
    """Base exception for all AgroVision ML errors."""
    pass


class ModelArtifactError(AgroVisionError):
    """Raised when model weights or artifact metadata cannot be loaded or are invalid."""
    pass


class ImageValidationError(AgroVisionError):
    """Raised when input image fails file format, corruption, or quality validation."""
    def __init__(self, reason: str, message: str, details: dict = None):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.details = details or {}


class InferenceError(AgroVisionError):
    """Raised when an unrecoverable error occurs during model forward pass or calibration."""
    pass

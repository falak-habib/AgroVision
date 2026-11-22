"""
AgroVision ML Inference Package.
Provides standalone, CPU-optimized crop disease screening and validation.
"""

from backend.ml.class_mapping import (
    CANONICAL_CLASSES,
    CLASS_REGISTRY,
    ClassInfo,
    get_all_classes,
    get_class_by_label,
    get_class_info,
    get_supported_crops,
)
from backend.ml.exceptions import (
    AgroVisionError,
    ImageValidationError,
    InferenceError,
    ModelArtifactError,
)
from backend.ml.image_validator import ImageValidator, ValidationResult
from backend.ml.model_loader import AgroVisionModelLoader, ModelMetadata
from backend.ml.predictor import CropDiseasePredictor
from backend.ml.preprocessing import ImagePreprocessor

__all__ = [
    "AgroVisionError",
    "ModelArtifactError",
    "ImageValidationError",
    "InferenceError",
    "ClassInfo",
    "CANONICAL_CLASSES",
    "CLASS_REGISTRY",
    "get_class_info",
    "get_class_by_label",
    "get_all_classes",
    "get_supported_crops",
    "AgroVisionModelLoader",
    "ModelMetadata",
    "ImagePreprocessor",
    "ImageValidator",
    "ValidationResult",
    "CropDiseasePredictor",
]

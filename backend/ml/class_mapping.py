"""
Class mapping utility for AgroVision 38-class plant pathology model.
Provides exact mapping from output indices (0-37) to canonical classes,
parsed crop species, disease conditions, and health status.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import re


@dataclass(frozen=True)
class ClassInfo:
    """Structured entity definition for a model class."""
    class_index: int
    canonical_label: str
    crop: str
    disease: str
    is_healthy: bool
    display_name: str


# Canonical 38-class list extracted directly from artifact.json
CANONICAL_CLASSES: List[str] = [
    "Apple__Apple_scab",
    "Apple__Black_rot",
    "Apple__Cedar_apple_rust",
    "Apple__healthy",
    "Blueberry__healthy",
    "Cherry_(including_sour)__Powdery_mildew",
    "Cherry_(including_sour)__healthy",
    "Corn_(maize)__Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)__Common_rust_",
    "Corn_(maize)__Northern_Leaf_Blight",
    "Corn_(maize)__healthy",
    "Grape__Black_rot",
    "Grape__Esca_(Black_Measles)",
    "Grape__Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape__healthy",
    "Orange__Haunglongbing_(Citrus_greening)",
    "Peach__Bacterial_spot",
    "Peach__healthy",
    "Pepper,_bell__Bacterial_spot",
    "Pepper,_bell__healthy",
    "Potato__Early_blight",
    "Potato__Late_blight",
    "Potato__healthy",
    "Raspberry__healthy",
    "Soybean__healthy",
    "Squash__Powdery_mildew",
    "Strawberry__Leaf_scorch",
    "Strawberry__healthy",
    "Tomato__Bacterial_spot",
    "Tomato__Early_blight",
    "Tomato__Late_blight",
    "Tomato__Leaf_Mold",
    "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites_Two-spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__healthy",
]


def _format_crop_name(raw_crop: str) -> str:
    """Format raw crop string into human-readable crop name."""
    clean = raw_crop.replace("_", " ")
    clean = clean.replace("(maize)", "").replace("(including sour)", "").strip()
    return clean


def _format_disease_name(raw_disease: str) -> str:
    """Format raw disease string into clean agronomic name."""
    if raw_disease.lower() == "healthy":
        return "Healthy"
    clean = raw_disease.replace("_", " ").strip()
    # Normalize common abbreviations or suffixes
    clean = re.sub(r"\s+", " ", clean)
    return clean


def _build_class_registry() -> Dict[int, ClassInfo]:
    """Build immutable dictionary mapping class index to ClassInfo."""
    registry = {}
    for idx, raw_label in enumerate(CANONICAL_CLASSES):
        parts = raw_label.split("__", 1)
        if len(parts) == 2:
            crop_part, disease_part = parts
        else:
            crop_part, disease_part = raw_label, "Unknown"

        is_healthy = disease_part.lower() == "healthy"
        crop_name = _format_crop_name(crop_part)
        disease_name = _format_disease_name(disease_part)
        
        if is_healthy:
            display = f"{crop_name} (Healthy)"
        else:
            display = f"{crop_name} - {disease_name}"

        registry[idx] = ClassInfo(
            class_index=idx,
            canonical_label=raw_label,
            crop=crop_name,
            disease=disease_name,
            is_healthy=is_healthy,
            display_name=display,
        )
    return registry


CLASS_REGISTRY: Dict[int, ClassInfo] = _build_class_registry()
LABEL_TO_INDEX: Dict[str, int] = {c.canonical_label: i for i, c in CLASS_REGISTRY.items()}


def get_class_info(index: int) -> ClassInfo:
    """Retrieve structured ClassInfo for a given integer index."""
    if index not in CLASS_REGISTRY:
        raise IndexError(f"Class index {index} out of range [0, {len(CLASS_REGISTRY)-1}]")
    return CLASS_REGISTRY[index]


def get_class_by_label(label: str) -> ClassInfo:
    """Retrieve structured ClassInfo for a given canonical label."""
    if label not in LABEL_TO_INDEX:
        raise KeyError(f"Label '{label}' not found in canonical class registry.")
    return CLASS_REGISTRY[LABEL_TO_INDEX[label]]


def get_all_classes() -> List[ClassInfo]:
    """Return all 38 ClassInfo objects ordered by class index."""
    return [CLASS_REGISTRY[i] for i in range(len(CLASS_REGISTRY))]


def get_supported_crops() -> List[str]:
    """Return sorted unique list of supported botanical crops."""
    crops = {info.crop for info in CLASS_REGISTRY.values()}
    return sorted(crops)

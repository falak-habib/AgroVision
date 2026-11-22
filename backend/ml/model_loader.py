"""
Model loader for AgroVision PyTorch MobileNetV2.
Handles artifact verification, architecture reconstruction, CPU loading,
and parameter freezing.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models

from backend.ml.exceptions import ModelArtifactError


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "ml" / "artifacts" / "all_v1"


@dataclass
class ModelMetadata:
    """Strongly typed container for model artifact metadata."""
    model_version: str
    arch: str
    classes: List[str]
    input_size: int
    normalize_mean: List[float]
    normalize_std: List[float]
    resize_strategy: str
    temperature: float
    abstain_threshold: float
    target_accepted_acc_on_val: Optional[float] = None
    torch_version_trained: Optional[str] = None


class AgroVisionModelLoader:
    """Manages loading, verification, and caching of the AgroVision PyTorch model."""

    def __init__(self, artifact_dir: Optional[Path] = None):
        self.artifact_dir = Path(artifact_dir) if artifact_dir else DEFAULT_ARTIFACT_DIR
        self.weights_path = self.artifact_dir / "model.pt"
        self.metadata_path = self.artifact_dir / "artifact.json"
        
        self.metadata: Optional[ModelMetadata] = None
        self._model: Optional[nn.Module] = None

    def load_metadata(self) -> ModelMetadata:
        """Load and validate artifact.json metadata."""
        if not self.metadata_path.exists():
            raise ModelArtifactError(
                f"Artifact metadata file not found at: {self.metadata_path}"
            )

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ModelArtifactError(f"Failed to parse artifact.json: {e}") from e

        required_keys = ["arch", "classes", "input_size", "normalize_mean", "normalize_std"]
        for k in required_keys:
            if k not in data:
                raise ModelArtifactError(f"Missing required key in artifact.json: '{k}'")

        if data["arch"] != "mobilenet_v2":
            raise ModelArtifactError(f"Unsupported architecture: '{data['arch']}'. Expected 'mobilenet_v2'.")

        self.metadata = ModelMetadata(
            model_version=data.get("model_version", "unknown"),
            arch=data["arch"],
            classes=data["classes"],
            input_size=int(data["input_size"]),
            normalize_mean=[float(x) for x in data["normalize_mean"]],
            normalize_std=[float(x) for x in data["normalize_std"]],
            resize_strategy=data.get("resize", "direct_resize_to_square"),
            temperature=float(data.get("temperature", 1.0)),
            abstain_threshold=float(data.get("abstain_threshold", 0.30)),
            target_accepted_acc_on_val=data.get("target_accepted_acc_on_val"),
            torch_version_trained=data.get("torch"),
        )
        return self.metadata

    def build_architecture(self, num_classes: int) -> nn.Module:
        """Construct MobileNetV2 architecture matching the exact training configuration."""
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    def load_model(self, force_reload: bool = False) -> Tuple[nn.Module, ModelMetadata]:
        """Load model weights onto CPU and set into eval mode."""
        if self._model is not None and not force_reload:
            return self._model, self.metadata

        if self.metadata is None or force_reload:
            self.load_metadata()

        if not self.weights_path.exists():
            raise ModelArtifactError(f"Model weights file not found at: {self.weights_path}")

        num_classes = len(self.metadata.classes)
        model = self.build_architecture(num_classes)

        try:
            # Force map_location to cpu for deterministic, standalone CPU serving
            state_dict = torch.load(self.weights_path, map_location=torch.device("cpu"))
        except Exception as e:
            raise ModelArtifactError(f"Failed to read weights from {self.weights_path}: {e}") from e

        # Strict state_dict key matching
        try:
            load_result = model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise ModelArtifactError(f"Weights parameter mismatch: {e}") from e

        # Set evaluation mode and freeze weights
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self._model = model
        return self._model, self.metadata

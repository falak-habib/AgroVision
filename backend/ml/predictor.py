"""
Core AgroVision crop disease predictor.
Provides unified, decoupled inference with pre-inference quality validation,
temperature-scaled confidence calibration, and selective classification (abstention).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import torch
import torch.nn.functional as F

from backend.ml.class_mapping import CLASS_REGISTRY, get_class_info
from backend.ml.exceptions import InferenceError
from backend.ml.image_validator import ImageValidator, ValidationResult
from backend.ml.model_loader import AgroVisionModelLoader, ModelMetadata
from backend.ml.preprocessing import ImagePreprocessor


class CropDiseasePredictor:
    """
    Production-oriented predictor encapsulating:
      1. One-time model and metadata loading onto CPU.
      2. Pre-inference image validation (blur, lighting, dimensions).
      3. Exact torchvision preprocessing transform.
      4. Forward pass with temperature calibration: softmax(logits / T).
      5. Abstention enforcement: confidence < threshold -> 'ABSTAIN'.
      6. Entity-resolved botanical classification output.
    """

    def __init__(
        self,
        artifact_dir: Optional[Union[str, Path]] = None,
        validator: Optional[ImageValidator] = None,
        custom_temperature: Optional[float] = None,
        custom_abstain_threshold: Optional[float] = None,
    ):
        self.loader = AgroVisionModelLoader(artifact_dir=Path(artifact_dir) if artifact_dir else None)
        self.model, self.metadata = self.loader.load_model()

        self.validator = validator if validator is not None else ImageValidator()
        self.preprocessor = ImagePreprocessor(
            img_size=self.metadata.input_size,
            normalize_mean=self.metadata.normalize_mean,
            normalize_std=self.metadata.normalize_std,
        )

        # Allow explicit override only for testing/experimentation, else use artifact values
        self.temperature = custom_temperature if custom_temperature is not None else self.metadata.temperature
        self.abstain_threshold = custom_abstain_threshold if custom_abstain_threshold is not None else self.metadata.abstain_threshold

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[Image.Image, bytes, bytearray, str, Path],
        validate: bool = True,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute full screening pipeline on an input image.

        Args:
            image_input: PIL Image, raw bytes, or filesystem path.
            validate: Whether to run pre-inference image quality and corruption checks.
            top_k: Number of top candidate classes to return.

        Returns:
            Dict containing status ('PREDICTION_AVAILABLE', 'ABSTAIN', or 'INVALID_IMAGE'),
            predicted entities, calibrated confidence score, and diagnostic metadata.
        """
        verified_img: Optional[Image.Image] = None
        validation_details: Dict[str, Any] = {}

        # Stage 1: Input Validation
        if validate:
            val_res, verified_img = self.validator.validate(image_input)
            if not val_res.is_valid:
                return {
                    "status": "INVALID_IMAGE",
                    "reason": val_res.reason,
                    "message": val_res.message,
                    "details": val_res.details,
                }
            validation_details = val_res.details

        # Stage 2: Preprocessing
        try:
            target_input = verified_img if verified_img is not None else image_input
            tensor = self.preprocessor.preprocess(target_input)
        except Exception as e:
            raise InferenceError(f"Image preprocessing failed: {e}") from e

        # Stage 3: Forward Pass & Temperature Calibration
        try:
            raw_logits = self.model(tensor)  # Shape: (1, 38)
            calibrated_logits = raw_logits / self.temperature
            probabilities = F.softmax(calibrated_logits, dim=1).squeeze(0)  # Shape: (38,)
        except Exception as e:
            raise InferenceError(f"Model forward pass failed: {e}") from e

        # Extract top-k candidates
        top_k = min(top_k, len(probabilities))
        top_probs, top_indices = torch.topk(probabilities, k=top_k)

        top_candidates: List[Dict[str, Any]] = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            info = get_class_info(idx)
            top_candidates.append({
                "class_index": idx,
                "canonical_class": info.canonical_label,
                "crop": info.crop,
                "disease": info.disease,
                "is_healthy": info.is_healthy,
                "display_name": info.display_name,
                "confidence": round(float(prob), 4),
            })

        best = top_candidates[0]
        confidence = best["confidence"]
        class_index = best["class_index"]

        # Stage 4: Abstention Decision Gating
        is_abstained = confidence < self.abstain_threshold

        status = "ABSTAIN" if is_abstained else "PREDICTION_AVAILABLE"
        message = (
            f"Confidence ({confidence*100:.1f}%) is below the diagnostic threshold ({self.abstain_threshold*100:.1f}%). "
            "Unable to reliably identify disease; flagged for specialist referral."
            if is_abstained
            else "Diagnosis successfully predicted."
        )

        return {
            "status": status,
            "message": message,
            "crop": best["crop"],
            "disease": best["disease"],
            "is_healthy": best["is_healthy"],
            "confidence": confidence,
            "class_index": class_index,
            "canonical_class": best["canonical_class"],
            "display_name": best["display_name"],
            "top_candidates": top_candidates,
            "abstain_threshold": self.abstain_threshold,
            "temperature_applied": round(self.temperature, 6),
            "validation_details": validation_details,
        }

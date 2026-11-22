"""
Image preprocessing pipeline for AgroVision ML inference.
Faithfully reproduces the exact evaluation transform defined in ml/train_agrovision.py:
  1. Image.convert("RGB")
  2. Resize((224, 224))
  3. ToTensor() -> scales to [0.0, 1.0]
  4. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
"""

import io
from pathlib import Path
from typing import List, Union
from PIL import Image
import torch
from torchvision import transforms

from backend.ml.exceptions import ImageValidationError


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
DEFAULT_IMG_SIZE = 224


class ImagePreprocessor:
    """Deterministic, production-ready preprocessor reproducing training evaluation transforms."""

    def __init__(
        self,
        img_size: int = DEFAULT_IMG_SIZE,
        normalize_mean: List[float] = None,
        normalize_std: List[float] = None,
    ):
        self.img_size = img_size
        self.mean = normalize_mean if normalize_mean is not None else DEFAULT_MEAN
        self.std = normalize_std if normalize_std is not None else DEFAULT_STD

        # Exact match to evalt transform in ml/train_agrovision.py
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def load_image(self, image_input: Union[Image.Image, bytes, str, Path]) -> Image.Image:
        """Coerce various input types into a standardized RGB PIL Image."""
        if isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, (bytes, bytearray)):
            try:
                img = Image.open(io.BytesIO(image_input))
            except Exception as e:
                raise ImageValidationError(
                    reason="INVALID_IMAGE_BYTES",
                    message="Failed to decode image from raw bytes.",
                    details={"error": str(e)}
                ) from e
        elif isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise ImageValidationError(
                    reason="FILE_NOT_FOUND",
                    message=f"Image file does not exist at: {path}",
                    details={"path": str(path)}
                )
            try:
                img = Image.open(path)
            except Exception as e:
                raise ImageValidationError(
                    reason="CORRUPT_IMAGE_FILE",
                    message=f"Failed to open image file at {path}.",
                    details={"error": str(e)}
                ) from e
        else:
            raise ImageValidationError(
                reason="UNSUPPORTED_INPUT_TYPE",
                message=f"Unsupported image input type: {type(image_input)}",
                details={"type": str(type(image_input))}
            )

        # Standardize color mode to 3-channel RGB
        try:
            return img.convert("RGB")
        except Exception as e:
            raise ImageValidationError(
                reason="RGB_CONVERSION_FAILED",
                message="Failed to convert image to RGB color space.",
                details={"error": str(e)}
            ) from e

    def preprocess(self, image_input: Union[Image.Image, bytes, str, Path]) -> torch.Tensor:
        """
        Preprocess an input image and return a 4D batch tensor of shape (1, 3, 224, 224).
        """
        img_rgb = self.load_image(image_input)
        tensor = self.transform(img_rgb)  # Shape: (3, 224, 224)
        return tensor.unsqueeze(0)        # Shape: (1, 3, 224, 224)

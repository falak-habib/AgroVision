"""
Input image validation module for AgroVision.
Performs pre-inference checks to intercept corrupt files, severe blur,
extreme exposure, and provides diagnostic leaf-presence heuristics.
"""

from dataclasses import dataclass, field
import io
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageOps


# Validation Thresholds
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_IMAGE_DIMENSION = 64                # At least 64x64 pixels
BLUR_LAPLACIAN_VAR_THRESHOLD = 50.0     # Below this is considered severely blurred
EXTREME_DARK_LUMINANCE_THRESHOLD = 25.0  # Mean luminance [0-255]
EXTREME_BRIGHT_LUMINANCE_THRESHOLD = 235.0 # Mean luminance [0-255]
HIGH_SATURATION_PIXEL_RATIO = 0.85       # Over 85% pixels at maximum intensity
SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "BMP"}


@dataclass
class ValidationResult:
    """Structured result returned by the image validator."""
    is_valid: bool
    reason: Optional[str] = None
    message: str = "Image passed validation checks."
    details: Dict[str, Union[float, int, str, bool]] = field(default_factory=dict)


class ImageValidator:
    """
    Validates input leaf photographs prior to model inference.
    Separates image quality validation from disease classification.
    """

    def __init__(
        self,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
        min_dimension: int = MIN_IMAGE_DIMENSION,
        blur_threshold: float = BLUR_LAPLACIAN_VAR_THRESHOLD,
        dark_threshold: float = EXTREME_DARK_LUMINANCE_THRESHOLD,
        bright_threshold: float = EXTREME_BRIGHT_LUMINANCE_THRESHOLD,
    ):
        self.max_file_size = max_file_size
        self.min_dimension = min_dimension
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold

    def validate(
        self,
        image_input: Union[Image.Image, bytes, bytearray, str, Path]
    ) -> Tuple[ValidationResult, Optional[Image.Image]]:
        """
        Execute full validation pipeline.
        Returns:
            Tuple[ValidationResult, Optional[Image.Image]]: Validation status and verified RGB image (if valid).
        """
        # Step 1: File size validation (if raw bytes or path provided)
        file_size_bytes = None
        if isinstance(image_input, (bytes, bytearray)):
            file_size_bytes = len(image_input)
        elif isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                return ValidationResult(
                    is_valid=False,
                    reason="FILE_NOT_FOUND",
                    message=f"Image file does not exist: {path}",
                ), None
            file_size_bytes = path.stat().st_size

        if file_size_bytes is not None and file_size_bytes > self.max_file_size:
            return ValidationResult(
                is_valid=False,
                reason="FILE_TOO_LARGE",
                message=f"Image file exceeds maximum allowable size of {self.max_file_size // (1024*1024)}MB.",
                details={"file_size_bytes": file_size_bytes, "max_allowed": self.max_file_size}
            ), None

        # Step 2: Image opening & corruption detection
        img: Image.Image
        if isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, (bytes, bytearray)):
            try:
                img = Image.open(io.BytesIO(image_input))
                img.verify()  # Check for integrity
                # Re-open because verify() consumes file pointer
                img = Image.open(io.BytesIO(image_input))
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    reason="CORRUPT_OR_UNREADABLE_IMAGE",
                    message="The uploaded file is corrupt or not a readable image.",
                    details={"error": str(e)}
                ), None
        elif isinstance(image_input, (str, Path)):
            try:
                img = Image.open(Path(image_input))
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    reason="CORRUPT_OR_UNREADABLE_IMAGE",
                    message="The image file could not be decoded.",
                    details={"error": str(e)}
                ), None
        else:
            return ValidationResult(
                is_valid=False,
                reason="UNSUPPORTED_INPUT_TYPE",
                message=f"Input type '{type(image_input).__name__}' is not supported.",
            ), None

        # Step 3: Format check
        if img.format and img.format.upper() not in SUPPORTED_FORMATS:
            return ValidationResult(
                is_valid=False,
                reason="UNSUPPORTED_IMAGE_FORMAT",
                message=f"Image format '{img.format}' is not supported. Use JPEG, PNG, WEBP, or BMP.",
                details={"detected_format": img.format}
            ), None

        # Step 4: Dimension check
        width, height = img.size
        if width < self.min_dimension or height < self.min_dimension:
            return ValidationResult(
                is_valid=False,
                reason="IMAGE_RESOLUTION_TOO_LOW",
                message=f"Image dimensions ({width}x{height}) are too small. Minimum required: {self.min_dimension}x{self.min_dimension}.",
                details={"width": width, "height": height, "min_required": self.min_dimension}
            ), None

        # Step 5: Convert to RGB
        try:
            img_rgb = img.convert("RGB")
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                reason="COLOR_CONVERSION_FAILED",
                message="Failed to convert image to RGB color space.",
                details={"error": str(e)}
            ), None

        # Step 6: Convert to numpy array for image quality analysis
        img_np = np.array(img_rgb)
        gray = np.array(ImageOps.grayscale(img_rgb), dtype=np.float32)

        mean_luminance = float(np.mean(gray))

        # Check for extreme darkness
        if mean_luminance < self.dark_threshold:
            return ValidationResult(
                is_valid=False,
                reason="IMAGE_TOO_DARK",
                message="The image is too dark to identify plant pathology. Please provide adequate illumination.",
                details={"mean_luminance": round(mean_luminance, 2), "threshold": self.dark_threshold}
            ), None

        # Check for extreme brightness / over-saturation
        saturated_pixel_ratio = float(np.mean(gray >= 250.0))
        if mean_luminance > self.bright_threshold or saturated_pixel_ratio > HIGH_SATURATION_PIXEL_RATIO:
            return ValidationResult(
                is_valid=False,
                reason="IMAGE_TOO_BRIGHT",
                message="The image is overexposed or washed out. Please reduce glare and direct flash.",
                details={
                    "mean_luminance": round(mean_luminance, 2),
                    "saturated_ratio": round(saturated_pixel_ratio, 3),
                    "threshold": self.bright_threshold
                }
            ), None

        # Step 7: Blur detection using discrete Laplacian variance
        laplacian_var = self._compute_laplacian_variance(gray)
        if laplacian_var < self.blur_threshold:
            return ValidationResult(
                is_valid=False,
                reason="IMAGE_TOO_BLURRY",
                message="The image is severely blurry. Please upload a clearer photograph of the leaf.",
                details={"laplacian_variance": round(laplacian_var, 2), "threshold": self.blur_threshold}
            ), None

        # Step 8: Diagnostic leaf presence analysis (heuristic only, does NOT reject)
        leaf_diag = self._analyze_leaf_heuristics(img_np)

        return ValidationResult(
            is_valid=True,
            reason=None,
            message="Image passed all quality and format validation checks.",
            details={
                "width": width,
                "height": height,
                "mean_luminance": round(mean_luminance, 2),
                "laplacian_variance": round(laplacian_var, 2),
                "leaf_presence_heuristic": leaf_diag,
            }
        ), img_rgb

    def _compute_laplacian_variance(self, gray: np.ndarray) -> float:
        """
        Compute variance of the Laplacian filter over the grayscale image.
        Uses a standard 3x3 discrete Laplace kernel [[0, 1, 0], [1, -4, 1], [0, 1, 0]].
        """
        h, w = gray.shape
        if h < 3 or w < 3:
            return 0.0

        # Convolve with discrete Laplacian kernel
        lap = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4.0 * gray[1:-1, 1:-1]
        )
        return float(np.var(lap))

    def _analyze_leaf_heuristics(self, img_np: np.ndarray) -> Dict[str, Union[float, str]]:
        """
        Compute color-based foliage heuristics (Excess Green Index 2G - R - B).
        CRITICAL ARCHITECTURAL NOTE:
        A low green score does NOT invalidate the image because heavily diseased leaves
        turn brown, necrotic, yellow, or purple. This score is diagnostic only.
        """
        r = img_np[:, :, 0].astype(np.float32)
        g = img_np[:, :, 1].astype(np.float32)
        b = img_np[:, :, 2].astype(np.float32)

        # Excess Green Index: 2G - R - B
        exg = 2.0 * g - r - b
        green_pixel_ratio = float(np.mean(exg > 15.0))

        # Yellow/Brown coverage (common in necrosis and chlorosis)
        # Yellow: R high, G high, B low (e.g. R>100, G>100, B<100, |R-G|<40)
        yellow_brown_ratio = float(np.mean((r > 80) & (g > 70) & (b < 120)))

        return {
            "green_pixel_ratio": round(green_pixel_ratio, 3),
            "yellow_brown_pixel_ratio": round(yellow_brown_ratio, 3),
            "note": "Diagnostic only. Rejection based on low green is prohibited to preserve sensitivity to necrotic disease."
        }

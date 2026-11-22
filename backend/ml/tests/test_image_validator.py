"""
Tests for ImageValidator:
Verifies input validation across corrupt files, invalid bytes, severe blur,
extreme darkness, extreme brightness, low resolution, and leaf heuristics.
"""

import io
import unittest
import numpy as np
from PIL import Image, ImageFilter

from backend.ml.image_validator import ImageValidator


class TestImageValidator(unittest.TestCase):
    def setUp(self):
        self.validator = ImageValidator(
            min_dimension=64,
            blur_threshold=50.0,
            dark_threshold=25.0,
            bright_threshold=235.0,
        )

    def test_valid_sharp_image(self):
        """A crisp, well-lit image with textured patterns should pass validation."""
        # Create a textured high-contrast image (e.g., checkerboard/edges)
        arr = np.zeros((200, 200, 3), dtype=np.uint8)
        arr[::4, :] = 200
        arr[:, ::4] = 180
        img = Image.fromarray(arr)
        
        result, verified_img = self.validator.validate(img)
        self.assertTrue(result.is_valid)
        self.assertIsNone(result.reason)
        self.assertIsNotNone(verified_img)

    def test_corrupted_bytes(self):
        """Random junk bytes should be rejected as corrupt or unreadable."""
        junk_bytes = b"This is not a valid JPEG or PNG header."
        result, img = self.validator.validate(junk_bytes)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.reason, "CORRUPT_OR_UNREADABLE_IMAGE")
        self.assertIsNone(img)

    def test_low_resolution_image(self):
        """An image below minimum dimensions (e.g. 32x32) must be rejected."""
        tiny_img = Image.new("RGB", (32, 32), color=(100, 150, 100))
        result, img = self.validator.validate(tiny_img)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.reason, "IMAGE_RESOLUTION_TOO_LOW")
        self.assertIn("dimensions", result.message)

    def test_severely_blurry_image(self):
        """A blurred image with minimal Laplacian variance must be rejected."""
        # Start with a sharp image, then apply strong Gaussian blur
        arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        sharp_img = Image.fromarray(arr)
        blurry_img = sharp_img.filter(ImageFilter.GaussianBlur(radius=15))

        result, _ = self.validator.validate(blurry_img)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.reason, "IMAGE_TOO_BLURRY")
        self.assertIn("clearer", result.message)

    def test_extremely_dark_image(self):
        """An almost pitch-black image (mean luminance < 25) must be rejected."""
        dark_img = Image.new("RGB", (200, 200), color=(10, 10, 10))
        result, _ = self.validator.validate(dark_img)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.reason, "IMAGE_TOO_DARK")
        self.assertIn("dark", result.message)

    def test_extremely_bright_image(self):
        """An overexposed, almost entirely white image must be rejected."""
        bright_img = Image.new("RGB", (200, 200), color=(250, 250, 250))
        result, _ = self.validator.validate(bright_img)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.reason, "IMAGE_TOO_BRIGHT")
        self.assertIn("overexposed", result.message)

    def test_leaf_presence_heuristic_does_not_reject_brown_leaf(self):
        """
        Verify that a necrotic/brown leaf (low green) is NOT rejected by quality checks.
        Validates rule: low green does not reject.
        """
        # Create a brown/necrotic textured leaf patch (R=120, G=70, B=30) with texture
        arr = np.zeros((200, 200, 3), dtype=np.uint8)
        arr[:, :] = [130, 80, 40]
        # Add high-contrast texture so it is not blurry
        arr[::5, :] = [180, 110, 50]
        arr[:, ::5] = [90, 50, 25]
        brown_leaf = Image.fromarray(arr)

        result, verified_img = self.validator.validate(brown_leaf)
        self.assertTrue(result.is_valid, "Necrotic brown leaf must NOT be rejected simply for low green.")
        self.assertIsNotNone(verified_img)
        self.assertIn("leaf_presence_heuristic", result.details)


if __name__ == "__main__":
    unittest.main()

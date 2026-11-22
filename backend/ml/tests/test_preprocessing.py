"""
Tests for ImagePreprocessor:
Verifies image loading, RGB conversion, resize to 224x224, tensor conversion,
and exact ImageNet normalization matching training.
"""

import unittest
import numpy as np
from PIL import Image
import torch

from backend.ml.preprocessing import ImagePreprocessor, DEFAULT_MEAN, DEFAULT_STD
from backend.ml.exceptions import ImageValidationError


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = ImagePreprocessor(img_size=224)

    def test_pil_image_preprocessing(self):
        """Verify tensor output shape and dtype from PIL Image."""
        img = Image.new("RGB", (300, 400), color=(100, 150, 200))
        tensor = self.preprocessor.preprocess(img)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_rgba_image_converted_to_rgb(self):
        """Verify 4-channel RGBA images are converted to 3-channel RGB."""
        img_rgba = Image.new("RGBA", (200, 200), color=(50, 100, 150, 255))
        tensor = self.preprocessor.preprocess(img_rgba)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_grayscale_image_converted_to_rgb(self):
        """Verify single-channel grayscale images are converted to 3-channel RGB."""
        img_gray = Image.new("L", (180, 180), color=128)
        tensor = self.preprocessor.preprocess(img_gray)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_normalization_values(self):
        """Verify that known pixel values produce the exact expected normalized output."""
        # Create a solid image of known RGB values
        r_val, g_val, b_val = 124, 116, 104
        img = Image.new("RGB", (224, 224), color=(r_val, g_val, b_val))
        tensor = self.preprocessor.preprocess(img)

        # Expected normalized value: ((val / 255.0) - mean) / std
        expected_r = ((r_val / 255.0) - DEFAULT_MEAN[0]) / DEFAULT_STD[0]
        expected_g = ((g_val / 255.0) - DEFAULT_MEAN[1]) / DEFAULT_STD[1]
        expected_b = ((b_val / 255.0) - DEFAULT_MEAN[2]) / DEFAULT_STD[2]

        self.assertAlmostEqual(tensor[0, 0, 100, 100].item(), expected_r, places=3)
        self.assertAlmostEqual(tensor[0, 1, 100, 100].item(), expected_g, places=3)
        self.assertAlmostEqual(tensor[0, 2, 100, 100].item(), expected_b, places=3)

    def test_invalid_input_type(self):
        """Verify error raised on invalid input type."""
        with self.assertRaises(ImageValidationError):
            self.preprocessor.preprocess(12345)


if __name__ == "__main__":
    unittest.main()

"""
Integration tests for CropDiseasePredictor:
Tests end-to-end inference flow, structured prediction outputs,
handling of invalid images, and abstention behavior.
"""

import unittest
import numpy as np
from PIL import Image

from backend.ml.predictor import CropDiseasePredictor


class TestPredictorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load predictor once for all integration tests
        cls.predictor = CropDiseasePredictor()

    def test_predict_sharp_synthetic_image(self):
        """End-to-end inference with a synthetic patterned image."""
        # Create a textured green leaf-like pattern
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[:, :, 1] = 160  # Green channel
        arr[::3, :, 0] = 60 # Red texture
        arr[:, ::3, 2] = 40 # Blue texture
        img = Image.fromarray(arr)

        result = self.predictor.predict(img)

        self.assertIn("status", result)
        self.assertIn(result["status"], ["PREDICTION_AVAILABLE", "ABSTAIN"])
        self.assertIn("crop", result)
        self.assertIn("disease", result)
        self.assertIn("confidence", result)
        self.assertIsInstance(result["confidence"], float)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertIn("class_index", result)
        self.assertGreaterEqual(result["class_index"], 0)
        self.assertLessEqual(result["class_index"], 37)
        self.assertEqual(len(result["top_candidates"]), 3)

    def test_predict_blurry_image_returns_invalid_image(self):
        """Predictor should intercept severely blurry image and return INVALID_IMAGE status."""
        # Create uniform/blurry image
        blurry_img = Image.new("RGB", (224, 224), color=(100, 150, 100))
        result = self.predictor.predict(blurry_img, validate=True)

        self.assertEqual(result["status"], "INVALID_IMAGE")
        self.assertEqual(result["reason"], "IMAGE_TOO_BLURRY")
        self.assertIn("blurry", result["message"].lower())

    def test_predict_abstention_enforcement(self):
        """Artificially high abstention threshold (e.g. 0.999) must trigger ABSTAIN status."""
        strict_predictor = CropDiseasePredictor(custom_abstain_threshold=0.999)
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[::2, ::2] = [40, 180, 50]
        img = Image.fromarray(arr)

        result = strict_predictor.predict(img)
        self.assertEqual(result["status"], "ABSTAIN")
        self.assertIn("specialist referral", result["message"].lower())


if __name__ == "__main__":
    unittest.main()

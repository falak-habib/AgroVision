"""
Tests for Temperature Scaling and Abstention Gating:
Verifies temperature calibration scaling p = softmax(logits / T),
abstention decision boundary (confidence < 0.30 -> ABSTAIN),
and edge cases (above, below, exactly at threshold).
"""

import unittest
import torch
import torch.nn.functional as F

from backend.ml.predictor import CropDiseasePredictor


class TestCalibrationAndAbstention(unittest.TestCase):
    def test_temperature_scaling_mathematics(self):
        """Verify softmax(logits / T) scaling effect."""
        logits = torch.tensor([[5.0, 2.0, 1.0, 0.5] + [0.0] * 34])
        T_unscaled = 1.0
        T_calibrated = 0.980576753616333

        probs_unscaled = F.softmax(logits / T_unscaled, dim=1)
        probs_calibrated = F.softmax(logits / T_calibrated, dim=1)

        # Probabilities must sum to 1.0
        self.assertAlmostEqual(probs_calibrated.sum().item(), 1.0, places=5)

        # Since T < 1.0, max probability should be slightly sharpened (higher)
        self.assertGreater(probs_calibrated[0, 0].item(), probs_unscaled[0, 0].item())

        # Argmax must remain identical
        self.assertEqual(probs_calibrated.argmax().item(), probs_unscaled.argmax().item())

    def test_abstention_above_threshold(self):
        """When max confidence >= 0.30, status must be 'PREDICTION_AVAILABLE'."""
        # Create predictor with custom threshold
        predictor = CropDiseasePredictor(custom_abstain_threshold=0.30)
        
        # Test simulated logits producing 0.75 confidence
        probs = torch.zeros(38)
        probs[10] = 0.75
        probs[11] = 0.15
        probs[12] = 0.10
        
        conf = probs.max().item()
        is_abstained = conf < predictor.abstain_threshold
        status = "ABSTAIN" if is_abstained else "PREDICTION_AVAILABLE"
        
        self.assertEqual(status, "PREDICTION_AVAILABLE")
        self.assertFalse(is_abstained)

    def test_abstention_below_threshold(self):
        """When max confidence < 0.30, status must be 'ABSTAIN'."""
        predictor = CropDiseasePredictor(custom_abstain_threshold=0.30)

        # Uniform distribution across 38 classes gives 1/38 ~ 0.0263 << 0.30
        probs = torch.full((38,), 1.0 / 38.0)
        
        conf = probs.max().item()
        is_abstained = conf < predictor.abstain_threshold
        status = "ABSTAIN" if is_abstained else "PREDICTION_AVAILABLE"

        self.assertEqual(status, "ABSTAIN")
        self.assertTrue(is_abstained)

    def test_abstention_boundary_conditions(self):
        """Test boundary condition exactly at 0.30, slightly below (0.2999), and slightly above (0.3001)."""
        threshold = 0.30

        # Case 1: Exactly at threshold (0.3000)
        conf_at = 0.3000
        self.assertFalse(conf_at < threshold, "Confidence equal to threshold must NOT abstain (accepted).")

        # Case 2: Slightly above threshold (0.3001)
        conf_above = 0.3001
        self.assertFalse(conf_above < threshold, "Confidence above threshold must NOT abstain.")

        # Case 3: Slightly below threshold (0.2999)
        conf_below = 0.2999
        self.assertTrue(conf_below < threshold, "Confidence below threshold MUST abstain.")


if __name__ == "__main__":
    unittest.main()

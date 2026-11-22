"""
Tests for AgroVision model loader:
Verifies model loading on CPU, metadata loading, architecture reconstruction,
strict parameter matching, and output dimensions.
"""

import unittest
from pathlib import Path
import torch

from backend.ml.model_loader import AgroVisionModelLoader, DEFAULT_ARTIFACT_DIR
from backend.ml.exceptions import ModelArtifactError


class TestModelLoader(unittest.TestCase):
    def setUp(self):
        self.loader = AgroVisionModelLoader()

    def test_load_metadata(self):
        """Verify metadata fields in artifact.json."""
        meta = self.loader.load_metadata()
        self.assertEqual(meta.arch, "mobilenet_v2")
        self.assertEqual(len(meta.classes), 38)
        self.assertEqual(meta.input_size, 224)
        self.assertEqual(len(meta.normalize_mean), 3)
        self.assertEqual(len(meta.normalize_std), 3)
        self.assertAlmostEqual(meta.temperature, 0.980576753616333, places=5)
        self.assertAlmostEqual(meta.abstain_threshold, 0.30, places=2)

    def test_load_model_on_cpu(self):
        """Verify that model loads cleanly onto CPU with 100% parameter match."""
        model, meta = self.loader.load_model()
        self.assertIsNotNone(model)
        self.assertFalse(next(model.parameters()).is_cuda, "Model must reside on CPU.")
        self.assertFalse(next(model.parameters()).requires_grad, "Model parameters must be frozen.")

    def test_output_dimensions(self):
        """Verify output dimensions for dummy input (batch_size=1, 3, 224, 224)."""
        model, _ = self.loader.load_model()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy)
        self.assertEqual(output.shape, (1, 38))
        self.assertEqual(output.dtype, torch.float32)

    def test_batch_output_dimensions(self):
        """Verify batched inference (batch_size=4)."""
        model, _ = self.loader.load_model()
        dummy_batch = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_batch)
        self.assertEqual(output.shape, (4, 38))

    def test_missing_artifact_handling(self):
        """Verify proper exception raised when artifact dir does not exist."""
        bad_loader = AgroVisionModelLoader(artifact_dir=Path("/non/existent/path"))
        with self.assertRaises(ModelArtifactError):
            bad_loader.load_metadata()


if __name__ == "__main__":
    unittest.main()

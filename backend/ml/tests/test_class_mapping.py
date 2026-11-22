"""
Tests for Class Mapping Module:
Verifies every class index from 0 to 37, ensures crop and disease parsing,
confirms 12 healthy classes, and cross-checks with artifact.json.
"""

import json
from pathlib import Path
import unittest

from backend.ml.class_mapping import (
    CANONICAL_CLASSES,
    CLASS_REGISTRY,
    get_all_classes,
    get_class_by_label,
    get_class_info,
    get_supported_crops,
)
from backend.ml.model_loader import DEFAULT_ARTIFACT_DIR


class TestClassMapping(unittest.TestCase):
    def test_canonical_classes_count(self):
        """Verify exactly 38 classes are present."""
        self.assertEqual(len(CANONICAL_CLASSES), 38)
        self.assertEqual(len(CLASS_REGISTRY), 38)

    def test_cross_check_with_artifact_json(self):
        """Verify CANONICAL_CLASSES matches artifact.json exactly in order and spelling."""
        artifact_path = DEFAULT_ARTIFACT_DIR / "artifact.json"
        with open(artifact_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(CANONICAL_CLASSES, data["classes"])

    def test_every_class_index_0_to_37(self):
        """Test retrieving every single class index from 0 to 37 without error."""
        for idx in range(38):
            info = get_class_info(idx)
            self.assertEqual(info.class_index, idx)
            self.assertIsInstance(info.crop, str)
            self.assertGreater(len(info.crop), 0)
            self.assertIsInstance(info.disease, str)
            self.assertGreater(len(info.disease), 0)
            self.assertIsInstance(info.is_healthy, bool)
            self.assertIsInstance(info.display_name, str)

    def test_healthy_classes_count(self):
        """Verify exactly 12 healthy classes exist across supported crops."""
        healthy_classes = [c for c in get_all_classes() if c.is_healthy]
        self.assertEqual(len(healthy_classes), 12)
        healthy_crops = {c.crop for c in healthy_classes}
        # Expected healthy crops
        expected_healthy = {
            "Apple", "Blueberry", "Cherry", "Corn", "Grape", "Peach",
            "Pepper, bell", "Potato", "Raspberry", "Soybean", "Strawberry", "Tomato"
        }
        self.assertEqual(healthy_crops, expected_healthy)

    def test_supported_crops_count(self):
        """Verify 14 unique crop species are supported."""
        crops = get_supported_crops()
        self.assertEqual(len(crops), 14)
        expected_crops = sorted([
            "Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach",
            "Pepper, bell", "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato"
        ])
        self.assertEqual(crops, expected_crops)

    def test_out_of_bounds_index(self):
        """Verify index out of bounds raises IndexError."""
        with self.assertRaises(IndexError):
            get_class_info(-1)
        with self.assertRaises(IndexError):
            get_class_info(38)

    def test_label_lookup(self):
        """Verify lookup by canonical string label."""
        info = get_class_by_label("Tomato__Early_blight")
        self.assertEqual(info.class_index, 29)
        self.assertEqual(info.crop, "Tomato")
        self.assertEqual(info.disease, "Early blight")
        self.assertFalse(info.is_healthy)


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for preprocessing modules.

Tests both:
- src/baseline/preprocessing.py  (grayscale + blur pipeline)
- src/proposed/preprocessing.py  (BGR colour pipeline)
"""
import sys
import os
import importlib.util

import numpy as np
import cv2
import pytest


def _load_module(name, path):
    """Load a Python file as a module by absolute path, bypassing the module cache."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Make same-directory imports (e.g. `from preprocessing import ...`) resolvable
    src_dir = os.path.dirname(path)
    sys.path.insert(0, src_dir)
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


_BASELINE_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'baseline')
_PROPOSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'proposed')

baseline_preprocessing = _load_module(
    "baseline_preprocessing",
    os.path.join(_BASELINE_DIR, "preprocessing.py")
)
proposed_preprocessing = _load_module(
    "proposed_preprocessing",
    os.path.join(_PROPOSED_DIR, "preprocessing.py")
)


class TestBaselinePreprocessSingleImage:
    def test_returns_2d_grayscale_array(self, synthetic_image_file):
        result = baseline_preprocessing.preprocess_single_image(synthetic_image_file)
        assert result is not None
        assert result.ndim == 2, "Baseline preprocessing should return a grayscale (2D) image"

    def test_output_matches_target_size(self, synthetic_image_file):
        target = (600, 300)
        result = baseline_preprocessing.preprocess_single_image(synthetic_image_file, target_size=target)
        # cv2 size is (width, height), numpy shape is (height, width)
        assert result.shape == (target[1], target[0])

    def test_custom_target_size(self, synthetic_image_file):
        target = (200, 100)
        result = baseline_preprocessing.preprocess_single_image(synthetic_image_file, target_size=target)
        assert result.shape == (100, 200)

    def test_invalid_path_returns_none(self):
        result = baseline_preprocessing.preprocess_single_image("/nonexistent/path/image.jpg")
        assert result is None

    def test_output_dtype_is_uint8(self, synthetic_image_file):
        result = baseline_preprocessing.preprocess_single_image(synthetic_image_file)
        assert result.dtype == np.uint8


class TestBaselineLoadDataset:
    def test_empty_directory_returns_empty_lists(self, tmp_path):
        images, labels = baseline_preprocessing.load_and_preprocess_dataset(str(tmp_path))
        assert images == []
        assert labels == []

    def test_loads_images_from_valid_class_folder(self, tmp_path, synthetic_bgr_image):
        class_dir = tmp_path / "idr_50000"
        class_dir.mkdir()
        img_path = str(class_dir / "test.jpg")
        cv2.imwrite(img_path, synthetic_bgr_image)

        images, labels = baseline_preprocessing.load_and_preprocess_dataset(str(tmp_path))
        assert len(images) == 1
        assert labels[0] == "idr_50000"

    def test_ignores_unknown_class_folders(self, tmp_path, synthetic_bgr_image):
        unknown_dir = tmp_path / "idr_999"
        unknown_dir.mkdir()
        cv2.imwrite(str(unknown_dir / "img.jpg"), synthetic_bgr_image)

        images, labels = baseline_preprocessing.load_and_preprocess_dataset(str(tmp_path))
        assert len(images) == 0


# ---------------------------------------------------------------------------
# Proposed preprocessing
# ---------------------------------------------------------------------------

class TestProposedPreprocessSingleImage:
    def test_returns_3d_bgr_array(self, synthetic_image_file):
        result = proposed_preprocessing.preprocess_single_image(synthetic_image_file)
        assert result is not None
        assert result.ndim == 3, "Proposed preprocessing should return a colour (3D) image"
        assert result.shape[2] == 3

    def test_output_matches_target_size(self, synthetic_image_file):
        target = (800, 400)
        result = proposed_preprocessing.preprocess_single_image(synthetic_image_file, target_size=target)
        assert result.shape == (target[1], target[0], 3)

    def test_custom_target_size(self, synthetic_image_file):
        target = (160, 80)
        result = proposed_preprocessing.preprocess_single_image(synthetic_image_file, target_size=target)
        assert result.shape == (80, 160, 3)

    def test_invalid_path_returns_none(self):
        result = proposed_preprocessing.preprocess_single_image("/nonexistent/image.jpg")
        assert result is None

    def test_output_dtype_is_uint8(self, synthetic_image_file):
        result = proposed_preprocessing.preprocess_single_image(synthetic_image_file)
        assert result.dtype == np.uint8


class TestProposedLoadDataset:
    def test_empty_directory_returns_empty_lists(self, tmp_path):
        images, labels = proposed_preprocessing.load_and_preprocess_dataset(str(tmp_path))
        assert images == []
        assert labels == []

    def test_loads_images_from_valid_class_folder(self, tmp_path, synthetic_bgr_image):
        class_dir = tmp_path / "idr_10000"
        class_dir.mkdir()
        cv2.imwrite(str(class_dir / "test.jpg"), synthetic_bgr_image)

        images, labels = proposed_preprocessing.load_and_preprocess_dataset(str(tmp_path))
        assert len(images) == 1
        assert labels[0] == "idr_10000"

    def test_multiple_classes_loaded(self, tmp_path, synthetic_bgr_image):
        for label in ["idr_1000", "idr_5000", "idr_50000"]:
            d = tmp_path / label
            d.mkdir()
            cv2.imwrite(str(d / "img.jpg"), synthetic_bgr_image)

        images, labels = proposed_preprocessing.load_and_preprocess_dataset(str(tmp_path))
        assert len(images) == 3
        assert set(labels) == {"idr_1000", "idr_5000", "idr_50000"}

    def test_returned_images_are_bgr(self, tmp_path, synthetic_bgr_image):
        d = tmp_path / "idr_20000"
        d.mkdir()
        cv2.imwrite(str(d / "img.jpg"), synthetic_bgr_image)

        images, _ = proposed_preprocessing.load_and_preprocess_dataset(str(tmp_path))
        assert images[0].ndim == 3

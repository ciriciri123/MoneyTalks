"""
Unit tests for feature extraction modules.

Tests both:
- src/baseline/features.py  (ORB only)
- src/proposed/features.py  (ORB + HSV colour)
"""
import sys
import os
import importlib.util
from unittest.mock import MagicMock

import numpy as np
import cv2
import pytest


def _load_module(name, path):
    """Load a Python file as a module by absolute path, bypassing the module cache."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    src_dir = os.path.dirname(path)
    sys.path.insert(0, src_dir)
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_kmeans(n_clusters=10):
    mock = MagicMock()
    mock.n_clusters = n_clusters
    mock.predict.return_value = np.random.randint(0, n_clusters, size=20)
    return mock


def _make_grayscale_image(h=300, w=600):
    """Synthetic grayscale image with enough texture for ORB keypoints."""
    np.random.seed(0)
    img = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (w - 20, h - 20), 180, -1)
    for i in range(8):
        cv2.circle(img, (60 + i * 70, h // 2), 25, 80 + i * 15, -1)
    noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


def _make_bgr_image(h=400, w=800):
    """Synthetic BGR image with enough texture for ORB keypoints."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (w - 30, h - 30), (200, 130, 60), -1)
    for i in range(10):
        cv2.circle(img, (50 + i * 75, h // 2), 20, (50 + i * 20, 100, 200 - i * 15), -1)
    noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


# ---------------------------------------------------------------------------
# Baseline features
# ---------------------------------------------------------------------------

_BASELINE_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'baseline')
_PROPOSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'proposed')

baseline_features = _load_module(
    "baseline_features",
    os.path.join(_BASELINE_DIR, "features.py")
)
proposed_features = _load_module(
    "proposed_features",
    os.path.join(_PROPOSED_DIR, "features.py")
)


class TestBaselineExtractOrbFeatures:
    def test_returns_one_descriptor_per_image(self):
        images = [_make_grayscale_image() for _ in range(3)]
        descriptors = baseline_features.extract_orb_features(images)
        assert len(descriptors) == 3

    def test_empty_image_list(self):
        descriptors = baseline_features.extract_orb_features([])
        assert descriptors == []

    def test_descriptor_dtype_is_uint8(self):
        images = [_make_grayscale_image()]
        descriptors = baseline_features.extract_orb_features(images)
        if len(descriptors[0]) > 0:
            assert descriptors[0].dtype == np.uint8

    def test_returns_empty_array_for_blank_image(self):
        blank = np.zeros((300, 600), dtype=np.uint8)
        descriptors = baseline_features.extract_orb_features([blank])
        assert len(descriptors) == 1  # still one entry, just empty


class TestBaselineBuildVisualVocabulary:
    def test_returns_kmeans_with_correct_n_clusters(self):
        images = [_make_grayscale_image() for _ in range(3)]
        desc_list = baseline_features.extract_orb_features(images)
        # Use small cluster count to keep the test fast
        kmeans = baseline_features.build_visual_vocabulary(desc_list, num_clusters=5)
        assert kmeans.n_clusters == 5

    def test_kmeans_has_cluster_centers(self):
        images = [_make_grayscale_image() for _ in range(3)]
        desc_list = baseline_features.extract_orb_features(images)
        kmeans = baseline_features.build_visual_vocabulary(desc_list, num_clusters=5)
        assert kmeans.cluster_centers_ is not None
        assert kmeans.cluster_centers_.shape[0] == 5


class TestBaselineExtractBovwHistograms:
    def test_output_shape_matches_images_and_clusters(self):
        n_images = 4
        n_clusters = 8
        mock_kmeans = _make_mock_kmeans(n_clusters)
        desc_list = [np.random.randint(0, 256, (20, 32), dtype=np.uint8) for _ in range(n_images)]
        histograms = baseline_features.extract_bovw_histograms(desc_list, mock_kmeans)
        assert histograms.shape == (n_images, n_clusters)

    def test_empty_descriptor_produces_zero_histogram(self):
        mock_kmeans = _make_mock_kmeans(8)
        desc_list = [np.array([])]
        histograms = baseline_features.extract_bovw_histograms(desc_list, mock_kmeans)
        assert np.all(histograms[0] == 0)

    def test_histograms_are_non_negative(self):
        mock_kmeans = _make_mock_kmeans(8)
        desc_list = [np.random.randint(0, 256, (20, 32), dtype=np.uint8)]
        histograms = baseline_features.extract_bovw_histograms(desc_list, mock_kmeans)
        assert np.all(histograms >= 0)


# ---------------------------------------------------------------------------
# Proposed features
# ---------------------------------------------------------------------------


class TestProposedExtractOrbAndColorFeatures:
    def test_returns_two_outputs_per_call(self):
        images = [_make_bgr_image()]
        desc_list, color_hists = proposed_features.extract_orb_and_color_features(images)
        assert len(desc_list) == 1
        assert color_hists.shape[0] == 1

    def test_color_histogram_length_is_512(self):
        images = [_make_bgr_image()]
        _, color_hists = proposed_features.extract_orb_and_color_features(images)
        assert color_hists.shape[1] == 512  # 8*8*8 bins

    def test_multiple_images(self):
        images = [_make_bgr_image() for _ in range(4)]
        desc_list, color_hists = proposed_features.extract_orb_and_color_features(images)
        assert len(desc_list) == 4
        assert color_hists.shape[0] == 4

    def test_color_histogram_is_normalised(self):
        images = [_make_bgr_image()]
        _, color_hists = proposed_features.extract_orb_and_color_features(images)
        norm = float(np.linalg.norm(color_hists[0]))
        assert abs(norm - 1.0) < 0.01, "HSV histogram should be L2-normalised"

    def test_empty_image_list(self):
        desc_list, color_hists = proposed_features.extract_orb_and_color_features([])
        assert desc_list == []
        assert color_hists.shape[0] == 0


class TestProposedBuildVisualVocabulary:
    def test_returns_kmeans_with_correct_n_clusters(self):
        images = [_make_bgr_image() for _ in range(3)]
        desc_list, _ = proposed_features.extract_orb_and_color_features(images)
        total_descs = sum(len(d) for d in desc_list if len(d) > 0)
        n_clusters = max(1, min(5, total_descs))
        kmeans = proposed_features.build_visual_vocabulary(desc_list, num_clusters=n_clusters)
        assert kmeans.n_clusters == n_clusters


class TestProposedExtractBovwHistograms:
    def test_output_shape_matches_images_and_clusters(self):
        n_images = 3
        n_clusters = 6
        mock_kmeans = _make_mock_kmeans(n_clusters)
        desc_list = [np.random.randint(0, 256, (20, 32), dtype=np.uint8) for _ in range(n_images)]
        histograms = proposed_features.extract_bovw_histograms(desc_list, mock_kmeans)
        assert histograms.shape == (n_images, n_clusters)

    def test_empty_descriptor_produces_zero_histogram(self):
        mock_kmeans = _make_mock_kmeans(6)
        histograms = proposed_features.extract_bovw_histograms([np.array([])], mock_kmeans)
        assert np.all(histograms[0] == 0)

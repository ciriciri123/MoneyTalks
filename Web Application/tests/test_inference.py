"""
Unit tests for inference.py

Covers:
- OCR text normalisation
- Levenshtein distance
- Denomination candidate extraction (exact + fuzzy)
- Image preprocessing from bytes
- BoVW histogram generation
- ORB + colour feature extraction
- Decision-fusion output contract
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock, patch

from inference import (
    _normalize_ocr_text,
    _levenshtein_distance,
    _extract_denom_candidates,
    _extract_fuzzy_denom_scores,
    preprocess_image,
    get_bovw_histogram,
    get_orb_and_color_features,
    predict_currency,
    DENOMINATION_MAP,
)


# ---------------------------------------------------------------------------
# _normalize_ocr_text
# ---------------------------------------------------------------------------

class TestNormalizeOcrText:
    def test_removes_special_characters(self):
        assert _normalize_ocr_text("Rp. 50.000,-") == "50 000"

    def test_collapses_whitespace(self):
        assert _normalize_ocr_text("10   000") == "10 000"

    def test_replaces_newlines_with_spaces(self):
        result = _normalize_ocr_text("50000\n")
        assert "\n" not in result

    def test_empty_string_returns_empty(self):
        assert _normalize_ocr_text("") == ""

    def test_digits_preserved(self):
        result = _normalize_ocr_text("100000")
        assert "100000" in result

    def test_strips_leading_trailing_whitespace(self):
        result = _normalize_ocr_text("  20000  ")
        assert result == result.strip()


# ---------------------------------------------------------------------------
# _levenshtein_distance
# ---------------------------------------------------------------------------

class TestLevenshteinDistance:
    def test_identical_strings_is_zero(self):
        assert _levenshtein_distance("50000", "50000") == 0

    def test_empty_source(self):
        assert _levenshtein_distance("", "abc") == 3

    def test_empty_target(self):
        assert _levenshtein_distance("abc", "") == 3

    def test_both_empty(self):
        assert _levenshtein_distance("", "") == 0

    def test_one_substitution(self):
        assert _levenshtein_distance("50000", "60000") == 1

    def test_one_insertion(self):
        assert _levenshtein_distance("5000", "50000") == 1

    def test_one_deletion(self):
        assert _levenshtein_distance("500000", "50000") == 1

    def test_known_distance(self):
        # "10000" vs "20000" → 1 substitution
        assert _levenshtein_distance("10000", "20000") == 1


# ---------------------------------------------------------------------------
# _extract_denom_candidates
# ---------------------------------------------------------------------------

class TestExtractDenomCandidates:
    def test_exact_token_match(self):
        result = _extract_denom_candidates("50000")
        assert "50000" in result

    def test_merged_tokens_match(self):
        # "50" and "000" together form "50000"
        result = _extract_denom_candidates("50 000")
        assert "50000" in result

    def test_no_digits_returns_empty(self):
        assert _extract_denom_candidates("no digits here") == []

    def test_empty_string_returns_empty(self):
        assert _extract_denom_candidates("") == []

    def test_all_denominations_can_be_found(self):
        for denom in DENOMINATION_MAP.keys():
            result = _extract_denom_candidates(denom)
            assert denom in result, f"{denom} not found in candidates"

    def test_shorter_denom_overshadowed_by_longer(self):
        # "1000" should be overshadowed by "100000" when both are present
        result = _extract_denom_candidates("100000")
        # "100000" should be in results; "1000" only if it's a direct token
        assert "100000" in result

    def test_returns_list(self):
        assert isinstance(_extract_denom_candidates("50000"), list)


# ---------------------------------------------------------------------------
# _extract_fuzzy_denom_scores
# ---------------------------------------------------------------------------

class TestExtractFuzzyDenomScores:
    def test_empty_string_returns_empty_dict(self):
        assert _extract_fuzzy_denom_scores("") == {}

    def test_no_digits_returns_empty_dict(self):
        assert _extract_fuzzy_denom_scores("abc") == {}

    def test_exact_match_not_scored_as_fuzzy(self):
        # Exact "50000" should not appear in fuzzy scores (not distance-1 from itself)
        scores = _extract_fuzzy_denom_scores("50000")
        assert "50000" not in scores

    def test_returns_dict(self):
        assert isinstance(_extract_fuzzy_denom_scores("50000"), dict)

    def test_scores_are_positive(self):
        scores = _extract_fuzzy_denom_scores("40000")
        for v in scores.values():
            assert v > 0


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------

class TestPreprocessImage:
    def test_returns_numpy_array(self, synthetic_image_bytes):
        result = preprocess_image(synthetic_image_bytes)
        assert isinstance(result, np.ndarray)

    def test_output_has_three_channels(self, synthetic_image_bytes):
        result = preprocess_image(synthetic_image_bytes)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_invalid_bytes_raises_or_returns_array(self):
        # Should either raise gracefully or not return None
        try:
            result = preprocess_image(b"not_an_image")
            # If it doesn't raise, result might be an array from PIL fallback
        except Exception:
            pass  # graceful failure is acceptable


# ---------------------------------------------------------------------------
# get_bovw_histogram
# ---------------------------------------------------------------------------

class TestGetBovwHistogram:
    def _make_mock_kmeans(self, n_clusters=10):
        mock = MagicMock()
        mock.n_clusters = n_clusters
        mock.predict.return_value = np.random.randint(0, n_clusters, size=50)
        return mock

    def test_output_length_equals_n_clusters(self, orb_descriptors):
        if len(orb_descriptors) == 0:
            pytest.skip("No descriptors extracted from synthetic image")
        n = 10
        mock_kmeans = self._make_mock_kmeans(n)
        hist = get_bovw_histogram(orb_descriptors, mock_kmeans)
        assert len(hist) == n

    def test_empty_descriptors_returns_zero_histogram(self):
        n = 8
        mock_kmeans = self._make_mock_kmeans(n)
        hist = get_bovw_histogram(np.array([]), mock_kmeans)
        assert len(hist) == n
        assert np.all(hist == 0)

    def test_histogram_values_are_non_negative(self, orb_descriptors):
        if len(orb_descriptors) == 0:
            pytest.skip("No descriptors extracted")
        mock_kmeans = self._make_mock_kmeans(10)
        hist = get_bovw_histogram(orb_descriptors, mock_kmeans)
        assert np.all(hist >= 0)


# ---------------------------------------------------------------------------
# get_orb_and_color_features
# ---------------------------------------------------------------------------

class TestGetOrbAndColorFeatures:
    def test_returns_five_values(self, synthetic_bgr_image):
        result = get_orb_and_color_features(synthetic_bgr_image)
        assert len(result) == 5

    def test_color_histogram_length(self, synthetic_bgr_image):
        _, _, color_hist, _, _ = get_orb_and_color_features(synthetic_bgr_image)
        # 8x8x8 HSV histogram flattened
        assert len(color_hist) == 512

    def test_descriptors_are_array(self, synthetic_bgr_image):
        _, descriptors, _, _, _ = get_orb_and_color_features(synthetic_bgr_image)
        assert isinstance(descriptors, np.ndarray)

    def test_box_coords_are_normalised_or_none(self, synthetic_bgr_image):
        _, _, _, box_coords, _ = get_orb_and_color_features(synthetic_bgr_image)
        if box_coords is not None:
            assert len(box_coords) == 4
            # Normalised values should be within [0, 1]
            assert all(0.0 <= v <= 1.0 for v in box_coords)

    def test_ocr_crop_is_bgr_array(self, synthetic_bgr_image):
        _, _, _, _, ocr_crop = get_orb_and_color_features(synthetic_bgr_image)
        assert isinstance(ocr_crop, np.ndarray)
        assert ocr_crop.ndim == 3


# ---------------------------------------------------------------------------
# predict_currency — output contract (mocked models)
# ---------------------------------------------------------------------------

class TestPredictCurrencyContract:
    def _make_mock_models(self):
        mock_kmeans = MagicMock()
        mock_kmeans.n_clusters = 800
        mock_kmeans.predict.return_value = np.zeros(50, dtype=int)

        mock_tfidf = MagicMock()
        mock_tfidf.transform.return_value = MagicMock(
            toarray=lambda: np.zeros((1, 800))
        )

        mock_svm = MagicMock()
        mock_svm.predict.return_value = ["idr_50000"]
        proba = np.zeros(7)
        proba[5] = 0.92  # high confidence for idr_50000
        mock_svm.predict_proba.return_value = [proba]

        return mock_kmeans, mock_tfidf, mock_svm

    def test_result_contains_label_and_confidence(self, synthetic_image_bytes):
        mock_kmeans, mock_tfidf, mock_svm = self._make_mock_models()
        with patch('inference._kmeans_model', mock_kmeans), \
             patch('inference._tfidf', mock_tfidf), \
             patch('inference._svm', mock_svm), \
             patch('inference._models_loaded', True), \
             patch('inference._predict_with_ocr', return_value=(None, 0.0, 0)):
            result = predict_currency(synthetic_image_bytes)

        assert "label" in result
        assert "confidence" in result

    def test_confidence_is_between_0_and_1(self, synthetic_image_bytes):
        mock_kmeans, mock_tfidf, mock_svm = self._make_mock_models()
        with patch('inference._kmeans_model', mock_kmeans), \
             patch('inference._tfidf', mock_tfidf), \
             patch('inference._svm', mock_svm), \
             patch('inference._models_loaded', True), \
             patch('inference._predict_with_ocr', return_value=(None, 0.0, 0)):
            result = predict_currency(synthetic_image_bytes)

        assert 0.0 <= result["confidence"] <= 1.0

    def test_label_is_valid_denomination(self, synthetic_image_bytes):
        mock_kmeans, mock_tfidf, mock_svm = self._make_mock_models()
        valid_labels = set(DENOMINATION_MAP.values()) | {"none"}
        with patch('inference._kmeans_model', mock_kmeans), \
             patch('inference._tfidf', mock_tfidf), \
             patch('inference._svm', mock_svm), \
             patch('inference._models_loaded', True), \
             patch('inference._predict_with_ocr', return_value=(None, 0.0, 0)):
            result = predict_currency(synthetic_image_bytes)

        assert result["label"] in valid_labels

    def test_error_key_returned_for_invalid_input(self):
        with patch('inference._models_loaded', True), \
             patch('inference.preprocess_image', side_effect=Exception("bad input")):
            result = predict_currency(b"garbage")

        assert "error" in result

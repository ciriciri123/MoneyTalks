import unittest
import io
import os
import sys
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services import inference
from app.services.inference import ModelNotLoadedError, ModelLoadError


class TestInference(unittest.TestCase):

    def setUp(self):
        inference._model = None
        inference._model_path = None

    def tearDown(self):
        inference._model = None
        inference._model_path = None

    def _create_dummy_image_bytes(self, width=10, height=10, color="red"):
        img = Image.new("RGB", (width, height), color=color)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def test_preprocess_success(self):
        image_bytes = self._create_dummy_image_bytes(width=100, height=100, color="blue")
        feature_vector = inference._preprocess(image_bytes)
        
        expected_shape = (224 * 224 * 3,)
        self.assertEqual(feature_vector.shape, expected_shape)
        self.assertEqual(feature_vector.dtype, np.float32)
        
        self.assertTrue(np.all(feature_vector >= 0.0))
        self.assertTrue(np.all(feature_vector <= 1.0))

    def test_preprocess_failure_invalid_bytes(self):
        with self.assertRaises(ValueError) as ctx:
            inference._preprocess(b"invalid image data")
        self.assertIn("Could not decode image bytes", str(ctx.exception))

    def test_predict_model_not_loaded(self):
        image_bytes = self._create_dummy_image_bytes()
        with self.assertRaises(ModelNotLoadedError):
            inference.predict(image_bytes)

    def test_predict_success(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = ["Rp 50.000"]
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9, 0.0]])

        inference._model = mock_model

        image_bytes = self._create_dummy_image_bytes()
        result = inference.predict(image_bytes)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["label"], "Rp 50.000")
        self.assertEqual(result["confidence"], 0.9)
        self.assertIn("inference_ms", result)
        self.assertIsInstance(result["inference_ms"], float)

        mock_model.predict.assert_called_once()
        mock_model.predict_proba.assert_called_once()

    def test_load_invalid_model_missing_methods(self):
        dummy_object = object()
        with patch("joblib.load", return_value=dummy_object):
            with self.assertRaises(ModelLoadError) as ctx:
                inference.load_model("dummy_path.pkl")
            self.assertIn("does not have predict() and predict_proba()", str(ctx.exception))

    def test_load_model_joblib_exception(self):
        with patch("joblib.load", side_effect=Exception("Failed to read file")):
            with self.assertRaises(ModelLoadError) as ctx:
                inference.load_model("corrupt.pkl")
            self.assertIn("Failed to load model", str(ctx.exception))

    def test_hot_swap_model_success(self):
        mock_model_1 = MagicMock()
        mock_model_1.predict.return_value = ["Rp 10.000"]
        mock_model_1.predict_proba.return_value = np.array([[0.95]])

        mock_model_2 = MagicMock()
        mock_model_2.predict.return_value = ["Rp 100.000"]
        mock_model_2.predict_proba.return_value = np.array([[0.99]])

        with patch("joblib.load", return_value=mock_model_1):
            inference.load_model("model1.pkl")
            self.assertEqual(inference._model, mock_model_1)

        with patch("joblib.load", return_value=mock_model_2):
            inference.hot_swap_model("model2.pkl")
            self.assertEqual(inference._model, mock_model_2)

        image_bytes = self._create_dummy_image_bytes()
        result = inference.predict(image_bytes)
        self.assertEqual(result["label"], "Rp 100.000")
        self.assertEqual(result["confidence"], 0.99)



if __name__ == "__main__":
    unittest.main()

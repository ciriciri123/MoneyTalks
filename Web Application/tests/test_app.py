"""
Unit tests for Flask app endpoints.

Uses Flask's built-in test client — no real server required.
"""
import sys
import os
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from unittest.mock import patch, MagicMock
from app import app


def _frame(image_bytes):
    """Wrap raw bytes in a BytesIO for Flask test client multipart upload."""
    return (io.BytesIO(image_bytes), 'frame.jpg')


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestIndexRoute:
    def test_returns_200(self, client):
        response = client.get('/')
        assert response.status_code == 200

    def test_returns_html(self, client):
        response = client.get('/')
        assert b'MoneyTalks' in response.data


# ---------------------------------------------------------------------------
# GET /test
# ---------------------------------------------------------------------------

class TestTestPageRoute:
    def test_returns_200(self, client):
        response = client.get('/test')
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/detect
# ---------------------------------------------------------------------------

class TestDetectEndpoint:
    def test_missing_frame_returns_400(self, client):
        response = client.post('/api/detect')
        assert response.status_code == 400

    def test_empty_frame_returns_400(self, client):
        data = {'frame': (b'', 'frame.jpg')}
        response = client.post('/api/detect', data=data, content_type='multipart/form-data')
        assert response.status_code == 400

    def test_valid_frame_high_confidence_returns_valid_true(self, client, synthetic_image_bytes):
        mock_result = {'label': 'idr_50000', 'confidence': 0.92}
        with patch('app.predict_currency', return_value=mock_result):
            data = {'frame': _frame(synthetic_image_bytes)}
            response = client.post('/api/detect', data=data, content_type='multipart/form-data')

        assert response.status_code == 200
        body = response.get_json()
        assert body['valid'] is True
        assert body['message'] == 'Lima Puluh Ribu Rupiah'

    def test_valid_frame_low_confidence_returns_valid_false(self, client, synthetic_image_bytes):
        mock_result = {'label': 'idr_50000', 'confidence': 0.40}
        with patch('app.predict_currency', return_value=mock_result):
            data = {'frame': _frame(synthetic_image_bytes)}
            response = client.post('/api/detect', data=data, content_type='multipart/form-data')

        assert response.status_code == 200
        body = response.get_json()
        assert body['valid'] is False
        assert 'dekatkan' in body['message'].lower()

    def test_inference_error_returns_500(self, client, synthetic_image_bytes):
        with patch('app.predict_currency', return_value={'error': 'something broke'}):
            data = {'frame': _frame(synthetic_image_bytes)}
            response = client.post('/api/detect', data=data, content_type='multipart/form-data')
        assert response.status_code == 500

    def test_response_includes_confidence(self, client, synthetic_image_bytes):
        mock_result = {'label': 'idr_10000', 'confidence': 0.85}
        with patch('app.predict_currency', return_value=mock_result):
            data = {'frame': _frame(synthetic_image_bytes)}
            response = client.post('/api/detect', data=data, content_type='multipart/form-data')
        body = response.get_json()
        assert 'confidence' in body

    def test_response_includes_bounding_box_when_present(self, client, synthetic_image_bytes):
        mock_result = {
            'label': 'idr_20000', 'confidence': 0.88,
            'box': [0.1, 0.2, 0.5, 0.4]
        }
        with patch('app.predict_currency', return_value=mock_result):
            data = {'frame': _frame(synthetic_image_bytes)}
            response = client.post('/api/detect', data=data, content_type='multipart/form-data')
        body = response.get_json()
        assert body['box'] == [0.1, 0.2, 0.5, 0.4]


# ---------------------------------------------------------------------------
# GET /api/tts
# ---------------------------------------------------------------------------

class TestTtsEndpoint:
    def test_missing_text_returns_400(self, client):
        response = client.get('/api/tts')
        assert response.status_code == 400

    def test_blank_text_returns_400(self, client):
        response = client.get('/api/tts?text=')
        assert response.status_code == 400

    def test_valid_text_returns_audio_mpeg(self, client):
        mock_audio = b'\xff\xfb\x90\x00' + b'\x00' * 100  # fake MP3 header
        mock_gtts = MagicMock()
        mock_gtts.write_to_fp = lambda fp: fp.write(mock_audio)

        with patch('app.gTTS', return_value=mock_gtts):
            response = client.get('/api/tts?text=Lima+Puluh+Ribu+Rupiah')

        assert response.status_code == 200
        assert response.content_type == 'audio/mpeg'

    def test_valid_text_returns_non_empty_body(self, client):
        mock_audio = b'\xff\xfb\x90\x00' + b'\x00' * 100
        mock_gtts = MagicMock()
        mock_gtts.write_to_fp = lambda fp: fp.write(mock_audio)

        with patch('app.gTTS', return_value=mock_gtts):
            response = client.get('/api/tts?text=Sepuluh+Ribu+Rupiah')

        assert len(response.data) > 0

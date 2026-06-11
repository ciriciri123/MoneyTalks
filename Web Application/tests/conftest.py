import pytest
import numpy as np
import cv2


@pytest.fixture
def synthetic_bgr_image():
    """800x400 BGR image with visible texture (not pure noise)."""
    np.random.seed(42)
    img = np.zeros((400, 800, 3), dtype=np.uint8)
    # Draw rectangles and lines so ORB can find real keypoints
    cv2.rectangle(img, (50, 50), (750, 350), (200, 150, 80), -1)
    cv2.rectangle(img, (100, 80), (700, 320), (120, 90, 40), 2)
    for i in range(10):
        x = 80 + i * 65
        cv2.circle(img, (x, 200), 20, (255, 255, 100), -1)
        cv2.putText(img, str(i * 10000), (x - 20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


@pytest.fixture
def synthetic_image_bytes(synthetic_bgr_image):
    """JPEG bytes of the synthetic BGR image."""
    _, buf = cv2.imencode('.jpg', synthetic_bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


@pytest.fixture
def synthetic_image_file(tmp_path, synthetic_bgr_image):
    """Write the synthetic image to a temp .jpg file and return the path string."""
    path = str(tmp_path / "test_banknote.jpg")
    cv2.imwrite(path, synthetic_bgr_image)
    return path


@pytest.fixture
def orb_descriptors(synthetic_bgr_image):
    """Extract real ORB descriptors from the synthetic image for use in feature tests."""
    orb = cv2.ORB_create(nfeatures=500)
    gray = cv2.cvtColor(synthetic_bgr_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, descriptors = orb.detectAndCompute(blurred, None)
    return descriptors if descriptors is not None else np.array([])

"""
app/services/inference.py — MoneyTalks ML Inference Engine
==========================================================
Manages the lifecycle of the scikit-learn .pkl model in memory:
    - load_model()      : Initial load at application startup
    - predict()         : Run inference on a JPEG frame (thread-safe)
    - hot_swap_model()  : Replace the in-memory model without restarting Flask

The model is stored as a module-level singleton guarded by a threading.Lock.
This is safe because:
    - Gunicorn workers are separate OS processes (no shared memory between them)
      so the lock only needs to protect against concurrent requests WITHIN one worker.
    - hot_swap_model() acquires the lock exclusively while replacing the model,
      so no request can read a half-replaced model object.

Called by:
    app/__init__.py      → load_model()
    app/routes.py        → predict() (per detection request)
    app/routes.py        → hot_swap_model() (on admin model deploy)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import joblib
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class ModelNotLoadedError(RuntimeError):
    """Raised when predict() is called before any model has been loaded."""
    pass


class ModelLoadError(ValueError):
    """Raised when a .pkl file cannot be deserialised or is incompatible."""
    pass


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================
# _model      : The loaded scikit-learn pipeline object (or None at startup)
# _model_path : The filesystem path of the currently loaded model file
# _lock       : threading.Lock protecting reads and writes to _model

_model:      Optional[Any] = None
_model_path: Optional[str] = None
_lock:       threading.Lock = threading.Lock()


# =============================================================================
# PUBLIC API
# =============================================================================

def load_model(model_path: str) -> None:
    """
    Load a .pkl model file into the module-level singleton.

    Called once by app/__init__.py during application startup.
    Acquires the lock for the duration of the load so that no
    concurrent request (in the same Gunicorn worker) can call
    predict() while the model object is being replaced.

    Args:
        model_path: Filesystem path to the .pkl file.

    Raises:
        ModelLoadError: If joblib.load() fails (corrupt file, incompatible
                        scikit-learn version, missing dependency).
    """
    global _model, _model_path

    logger.info("Loading model from: %s", model_path)
    start = time.perf_counter()

    try:
        candidate = joblib.load(model_path)
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load model from '{model_path}': {exc}"
        ) from exc

    # Validate that the object looks like a scikit-learn estimator
    if not hasattr(candidate, "predict") and not hasattr(candidate, "predict_proba"):
        raise ModelLoadError(
            f"The object loaded from '{model_path}' does not have predict() "
            "and predict_proba() methods. Ensure it is a fitted scikit-learn estimator."
        )

    elapsed = (time.perf_counter() - start) * 1000
    with _lock:
        _model      = candidate
        _model_path = model_path

    logger.info("Model loaded in %.1f ms. Type: %s", elapsed, type(_model).__name__)


def hot_swap_model(model_path: str) -> None:
    """
    Replace the in-memory model with a newly loaded one — zero downtime.

    Called by the admin deploy route (routes.py) after:
        1. The DB flag has been updated in Supabase.
        2. The new .pkl bytes have been written to model_path on disk.

    The swap is atomic from the perspective of other threads:
        - The new model is fully loaded (joblib.load) BEFORE acquiring the lock.
        - The lock is held only for the variable assignment (~microseconds).
        - Any request that is currently inside predict() with the old model
          will complete normally. The next request will get the new model.

    Args:
        model_path: Filesystem path to the new .pkl file.

    Raises:
        ModelLoadError: If the new model file cannot be loaded.
    """
    logger.info("Hot-swapping model from: %s", model_path)

    # Load the candidate OUTSIDE the lock to minimise lock-hold time
    try:
        candidate = joblib.load(model_path)
    except Exception as exc:
        raise ModelLoadError(f"Hot-swap failed to load '{model_path}': {exc}") from exc

    if not hasattr(candidate, "predict") or not hasattr(candidate, "predict_proba"):
        raise ModelLoadError("Hot-swap candidate is not a valid scikit-learn estimator.")

    global _model, _model_path
    with _lock:
        old_type    = type(_model).__name__ if _model else "None"
        _model      = candidate
        _model_path = model_path

    logger.info("Model hot-swapped. Old: %s → New: %s", old_type, type(_model).__name__)


def predict(image_bytes: bytes) -> dict:
    """
    Pre-process a JPEG frame and run inference against the in-memory model.

    This function is called on every video frame that reaches /api/detect.
    It must be fast: target < 2 seconds total round-trip.

    Thread safety:
        Acquires _lock in shared-read mode (effectively — Python's GIL combined
        with the assignment atomicity of CPython means concurrent reads of _model
        are safe even without explicit locking). The lock is only strictly needed
        during hot_swap_model() writes, which is already handled there.

    Args:
        image_bytes: Raw JPEG bytes from the browser canvas.toBlob() call.

    Returns:
        A dict with keys:
            label        (str):   Predicted denomination, e.g. 'Rp 50.000'.
            confidence   (float): Probability of the predicted class (0.0–1.0).
            inference_ms (float): Time spent inside model.predict(), in ms.

    Raises:
        ModelNotLoadedError: If load_model() has not been called yet.
        ValueError:          If the image cannot be decoded or pre-processed.
    """
    if _model is None:
        raise ModelNotLoadedError("No model is loaded. Call load_model() first.")

    # ── Pre-processing ─────────────────────────────────────────────────────────
    # Convert JPEG bytes → PIL Image → NumPy array.
    # The exact operations here (resize dimensions, colour mode) must match
    # what was used during model training. Update these constants to match
    # the ML team's pipeline specification.
    feature_vector = _preprocess(image_bytes)

    # ── Inference ──────────────────────────────────────────────────────────────
    start = time.perf_counter()

    # Snapshot the model reference under the lock to guard against a hot-swap
    # occurring between the None-check above and the predict() call below.
    with _lock:
        model_snapshot = _model

    # predict() and predict_proba() are called outside the lock so concurrent
    # requests in the same process are not serialised.
    label_array  = model_snapshot.predict([feature_vector])
    proba_array  = model_snapshot.predict_proba([feature_vector])

    elapsed_ms = (time.perf_counter() - start) * 1000

    predicted_label: str   = str(label_array[0])
    confidence:      float = float(np.max(proba_array[0]))

    logger.debug("Inference: label=%s confidence=%.4f time=%.1fms",
                 predicted_label, confidence, elapsed_ms)

    return {
        "label":        predicted_label,
        "confidence":   confidence,
        "inference_ms": round(elapsed_ms, 2),
    }


# =============================================================================
# PRIVATE: PRE-PROCESSING
# =============================================================================

# ── CONFIGURATION — match these to the ML team's training pipeline ─────────────
# Ask the ML team: "What input shape and preprocessing did you use when training?"
_TARGET_SIZE  = (224, 224)   # resize target (width, height) in pixels
_COLOUR_MODE  = "RGB"        # "RGB" or "L" (grayscale)
_FLATTEN      = True         # True if the model expects a 1-D feature vector


def _preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw JPEG bytes into the NumPy array shape expected by the model.

    Steps:
        1. Decode JPEG bytes into a PIL Image.
        2. Convert colour mode (e.g. RGBA → RGB, or RGB → grayscale).
        3. Resize to the target dimensions.
        4. Convert to float32 NumPy array.
        5. Normalise pixel values to [0, 1].
        6. Flatten to a 1-D vector if the model expects flat features.

    Args:
        image_bytes: Raw JPEG bytes.

    Returns:
        A NumPy array ready to pass as a single sample to model.predict().

    Raises:
        ValueError: If the bytes cannot be decoded as an image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise ValueError(f"Could not decode image bytes: {exc}") from exc

    img = img.convert(_COLOUR_MODE)
    img = img.resize(_TARGET_SIZE, Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)
    arr = arr / 255.0          # normalise to [0, 1]

    if _FLATTEN:
        arr = arr.flatten()    # shape: (width * height * channels,)

    return arr

"""
app/routes.py — MoneyTalks Guest & API Routes
=============================================
This module defines two categories of routes:

1. Guest routes (no authentication required):
       GET  /                  — Renders the banknote detection page
       POST /api/detect        — Accepts a video frame, returns detection result
       POST /api/upload-image  — Persists a detected frame to Supabase
       GET  /api/tts           — gTTS fallback: returns spoken audio as MP3

2. Admin API routes (session required, exposed via auth_bp in auth.py):
       GET  /admin/images/download    — Streams a zip of scanned images
       POST /admin/models/upload      — Uploads a new .pkl model file
       PATCH /admin/models/<id>/deploy — Activates a model version (hot-swap)

Architecture note:
    Routes are thin orchestrators — they validate input, call one service
    function, and return a response. All business logic lives in:
        app/services/supabase_client.py  (data persistence)
        app/services/inference.py        (model prediction)
        app/auth.py                      (authentication)
"""

import io
import logging
import os
import pathlib
import uuid as uuid_module
from datetime import date

import bcrypt
from flask import (
    Blueprint, current_app, jsonify, redirect,
    render_template, request, send_file, url_for,
)
from flask_login import current_user, login_required
from werkzeug.exceptions import BadRequest

from app import csrf, limiter
from app.services.supabase_client import get_db, SupabaseError
from app.services import inference

logger   = logging.getLogger(__name__)
main_bp  = Blueprint("main", __name__)


# =============================================================================
# GUEST ROUTES
# =============================================================================

@main_bp.route("/")
def index():
    """
    GET / — Render the guest banknote detection page.

    Passes the capture interval and confidence threshold from config into
    the Jinja2 template so the JavaScript layer uses consistent values
    without hardcoding them in the browser.
    """
    return render_template(
        "guest/index.html",
        capture_interval_ms  = current_app.config["CAPTURE_INTERVAL_MS"],
        confidence_threshold = current_app.config["CONFIDENCE_THRESHOLD"],
    )


@main_bp.route("/api/detect", methods=["POST"])
@csrf.exempt        # multipart/form-data from JS fetch(); CSRF token not included
@limiter.limit("60 per minute")   # anti-abuse: ~1 req/sec with some headroom
def api_detect():
    """
    POST /api/detect — Run the ML model on an uploaded video frame.

    Request:
        Multipart/form-data with one field:
            frame (file): JPEG image bytes from the browser canvas.

    Response (JSON):
        On success (confidence >= threshold):
            { "detected": true, "label": "Rp 50.000", "confidence": 0.94,
              "inference_ms": 312 }

        On low confidence:
            { "detected": false, "message": "Posisikan uang lebih dekat ke kamera",
              "confidence": 0.41 }

        On error:
            HTTP 400 or 500 with { "error": "description" }

    The caller (browser JS) handles the response: updates the UI,
    triggers TTS, and fires the upload-image request if detected=true.
    """
    if "frame" not in request.files:
        return jsonify({"error": "No frame file in request."}), 400

    frame_file = request.files["frame"]
    image_bytes = frame_file.read()

    if not image_bytes:
        return jsonify({"error": "Empty frame received."}), 400

    # Run inference (may raise if model is not loaded)
    try:
        result = inference.predict(image_bytes)
    except inference.ModelNotLoadedError:
        logger.error("/api/detect called but no model is loaded.")
        return jsonify({"error": "Model not available. Please contact the administrator."}), 503
    except Exception as exc:
        logger.exception("Unexpected inference error: %s", exc)
        return jsonify({"error": "Inference failed. Please try again."}), 500

    threshold = current_app.config["CONFIDENCE_THRESHOLD"]

    if result["confidence"] < threshold:
        return jsonify({
            "detected":   False,
            "message":    "Posisikan uang lebih dekat ke kamera",
            "confidence": round(result["confidence"], 4),
        })

    return jsonify({
        "detected":      True,
        "label":         result["label"],
        "confidence":    round(result["confidence"], 4),
        "inference_ms":  result["inference_ms"],
    })


@main_bp.route("/api/upload-image", methods=["POST"])
@csrf.exempt        # called by browser JS fetch() after successful detection
@limiter.limit("60 per minute")
def api_upload_image():
    """
    POST /api/upload-image — Persist a detected banknote frame to Supabase.

    This endpoint is called asynchronously by the browser after /api/detect
    returns detected=true. It is fire-and-forget from the browser's perspective:
    the detection cycle continues regardless of whether this upload succeeds.

    Request:
        Multipart/form-data with:
            frame          (file): JPEG bytes of the detected frame.
            detected_label (str):  Denomination label from the prior /api/detect call.

    Response:
        HTTP 202 Accepted (always, even on upload failure — so the browser
        detection loop is never blocked by a persistence error).

    Failures are logged server-side. The retry logic in supabase_client.py
    handles transient Supabase errors.
    """
    if "frame" not in request.files or "detected_label" not in request.form:
        return jsonify({"error": "Missing frame or detected_label."}), 400

    image_bytes     = request.files["frame"].read()
    detected_label  = request.form["detected_label"].strip()

    if not image_bytes or not detected_label:
        return jsonify({"error": "Empty frame or label."}), 400

    try:
        record = get_db().upload_scan_image(image_bytes, detected_label)
        logger.debug("Upload persisted: image_id=%s", record.image_id)
    except SupabaseError as exc:
        # Log but return 202 so the browser is not blocked
        logger.error("upload-image Supabase error (non-blocking): %s", exc)

    # Always 202: the browser should not retry on failure (supabase_client.py
    # handles retries internally).
    return jsonify({"status": "accepted"}), 202


@main_bp.route("/api/tts")
@limiter.limit("30 per minute")
def api_tts():
    """
    GET /api/tts?label=<text> — gTTS fallback TTS endpoint.

    Called ONLY when the browser's Web Speech API is unavailable.
    Generates an MP3 from the denomination label using gTTS and streams it.

    Query parameters:
        label (str): The denomination text to speak, e.g. 'Rp 50.000'.

    Response:
        audio/mpeg stream that the browser plays via an <audio> element.

    Requires outbound internet access from the server to Google's TTS API.
    """
    try:
        from gtts import gTTS
    except ImportError:
        return jsonify({"error": "gTTS not installed on this server."}), 501

    label = request.args.get("label", "").strip()
    if not label:
        return jsonify({"error": "Missing label parameter."}), 400

    try:
        tts_obj = gTTS(text=label, lang="id", slow=False)
        mp3_buf = io.BytesIO()
        tts_obj.write_to_fp(mp3_buf)
        mp3_buf.seek(0)
        return send_file(mp3_buf, mimetype="audio/mpeg", as_attachment=False)
    except Exception as exc:
        logger.error("gTTS generation failed for label '%s': %s", label, exc)
        return jsonify({"error": "TTS generation failed."}), 500


# =============================================================================
# ADMIN API ROUTES (protected, registered via auth_bp but defined here for
# organisation — import and register them in auth.py with url_prefix="/admin")
# =============================================================================

@main_bp.route("/admin/images/download")
@login_required
def admin_download_images():
    """
    GET /admin/images/download — Stream a ZIP archive of scanned images.

    Query parameters:
        from (str, YYYY-MM-DD): Start of date range.
        to   (str, YYYY-MM-DD): End of date range.

    Response:
        application/zip stream with filename scanned-images-<from>-<to>.zip
    """
    def _parse(param):
        try:
            return date.fromisoformat(param) if param else None
        except ValueError:
            return None

    from_date = _parse(request.args.get("from"))
    to_date   = _parse(request.args.get("to"))

    try:
        zip_bytes = get_db().download_scans_as_zip(from_date=from_date, to_date=to_date)
    except SupabaseError as exc:
        logger.error("Image download failed: %s", exc)
        return jsonify({"error": "Failed to build zip archive."}), 500

    from_str = from_date.isoformat() if from_date else "all"
    to_str   = to_date.isoformat()   if to_date   else "all"
    filename = f"scanned-images-{from_str}-{to_str}.zip"

    return send_file(
        io.BytesIO(zip_bytes),
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename,
    )


@main_bp.route("/admin/models/upload", methods=["POST"])
@login_required
def admin_upload_model():
    """
    POST /admin/models/upload — Upload a new .pkl model file.

    Request (multipart/form-data):
        model_file (file):  The .pkl file.
        version    (str):   Semantic version string, e.g. '2.1.0'.

    Validates:
        - File extension must be .pkl
        - File must be non-empty
        - Version string must be non-empty
        - File size checked implicitly by Flask's MAX_CONTENT_LENGTH

    Response:
        Redirects to /admin/models on success.
        Returns 400/500 JSON on failure.
    """
    if "model_file" not in request.files:
        return jsonify({"error": "No model_file in request."}), 400

    model_file = request.files["model_file"]
    version    = request.form.get("version", "").strip()

    if not version:
        return jsonify({"error": "Version string is required."}), 400
    if not model_file.filename.endswith(".pkl"):
        return jsonify({"error": "Only .pkl files are accepted."}), 400

    pkl_bytes = model_file.read()
    if not pkl_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    # Generate a short unique model_id: MDL- + first 8 chars of a UUID
    model_id = "MDL-" + str(uuid_module.uuid4()).replace("-", "")[:8].upper()

    try:
        get_db().upload_model_file(
            model_id  = model_id,
            version   = version,
            admin_id  = current_user.admin_id,
            pkl_bytes = pkl_bytes,
        )
        logger.info("Admin %s uploaded model %s (version %s)",
                    current_user.admin_id, model_id, version)
    except SupabaseError as exc:
        logger.error("Model upload failed: %s", exc)
        return jsonify({"error": "Model upload failed. See server logs."}), 500

    return redirect(url_for("auth.models"))


@main_bp.route("/admin/models/<model_id>/deploy", methods=["POST"])
@login_required
def admin_deploy_model(model_id: str):
    """
    POST /admin/models/<model_id>/deploy — Activate a model version.

    HTML forms cannot send PATCH requests. We use POST with a hidden
    _method field for semantic clarity, but the route accepts POST.

    Flow:
        1. Update ModelVersions.is_deployed in Supabase (DB trigger handles cleanup).
        2. Download the new .pkl bytes from Supabase Storage.
        3. Write bytes to local MODEL_PATH cache.
        4. Call inference.hot_swap_model() to replace in-memory model object.

    Response:
        Redirects to /admin/models on success.
        Returns 500 JSON on failure.
    """
    model_path = current_app.config["MODEL_PATH"]

    try:
        # Step 1: flip the DB flag
        model_record = get_db().deploy_model(model_id)

        # Step 2: download the new .pkl
        pkl_bytes = get_db().download_model_bytes_by_id(model_id)

        # Step 3: write to local cache
        pathlib.Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(pkl_bytes)
        logger.info("Model %s (%s) written to local cache: %s",
                    model_id, model_record.version, model_path)

        # Step 4: hot-swap in-memory model (thread-safe)
        inference.hot_swap_model(model_path)
        logger.info("Model hot-swapped to version %s", model_record.version)

    except SupabaseError as exc:
        logger.error("Deploy model %s failed: %s", model_id, exc)
        return jsonify({"error": f"Deployment failed: {exc}"}), 500
    except inference.ModelLoadError as exc:
        logger.error("inference.hot_swap_model failed for %s: %s", model_id, exc)
        return jsonify({"error": "Model file is invalid or incompatible."}), 500

    return redirect(url_for("auth.models"))

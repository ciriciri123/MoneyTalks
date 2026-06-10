"""
config.py — MoneyTalks Application Configuration
=================================================
Provides three configuration classes that Flask's application factory
(app/__init__.py) selects from based on the FLASK_ENV environment variable:

    development  →  DevelopmentConfig
    testing      →  TestingConfig
    production   →  ProductionConfig  (default)

All secrets are read from environment variables — never hardcoded here.
In local development, python-dotenv automatically loads them from .env.
In production (Docker), they are injected via docker-compose environment blocks.

Usage inside the app factory:
    app.config.from_object(get_config())
"""

import os
from datetime import timedelta


def get_config() -> "BaseConfig":
    """
    Return the correct config class based on the FLASK_ENV environment variable.
    Defaults to ProductionConfig if FLASK_ENV is not set.

    Called once by the application factory in app/__init__.py.
    """
    env = os.getenv("FLASK_ENV", "production").lower()
    mapping = {
        "development": DevelopmentConfig,
        "testing":     TestingConfig,
        "production":  ProductionConfig,
    }
    return mapping.get(env, ProductionConfig)


# =============================================================================
# BASE CONFIGURATION
# =============================================================================

class BaseConfig:
    """
    Shared settings inherited by all environment-specific configs.

    All values that reference os.getenv() are evaluated at import time,
    so the .env file must be loaded BEFORE this module is imported.
    python-dotenv's load_dotenv() handles this in run.py.
    """

    # ── Flask core ────────────────────────────────────────────────────────────
    # SECRET_KEY signs session cookies and CSRF tokens.
    # Must be a cryptographically random string; 32+ bytes recommended.
    # Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
    SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")

    # ── Session settings ──────────────────────────────────────────────────────
    # Sessions are server-side (Flask-Login); only a session ID is in the cookie.
    SESSION_COOKIE_HTTPONLY: bool = True    # JS cannot read the session cookie
    SESSION_COOKIE_SECURE: bool = True      # Cookie only sent over HTTPS
    SESSION_COOKIE_SAMESITE: str = "Lax"   # Protects against CSRF while allowing normal nav
    PERMANENT_SESSION_LIFETIME: timedelta = timedelta(minutes=30)  # Admin session timeout

    # ── Supabase connection ───────────────────────────────────────────────────
    # SUPABASE_URL  : Your project URL, e.g. https://abcdefgh.supabase.co
    # SUPABASE_KEY  : The SERVICE ROLE key — NOT the anon key.
    #                 The service role key bypasses RLS and must NEVER be
    #                 exposed to the browser or committed to version control.
    # Both values are found in: Supabase Dashboard → Project Settings → API
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY", "")

    # ── Supabase Storage bucket names ─────────────────────────────────────────
    # These must exactly match the bucket names you created in the Supabase Dashboard.
    SUPABASE_IMAGES_BUCKET: str = "scanned-images"
    SUPABASE_MODELS_BUCKET: str = "model-files"

    # ── ML model ──────────────────────────────────────────────────────────────
    # Local filesystem path where the active .pkl is cached after download.
    # The models/ directory is outside app/ so it is not served as a static file.
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/active/model.pkl")

    # Minimum model.predict_proba() confidence score to accept a detection.
    # Frames below this threshold are discarded and no image is uploaded.
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

    # ── Camera / detection ────────────────────────────────────────────────────
    # How often (ms) the browser captures and submits a frame.
    # Sent to the Jinja2 template as a JS variable; does not affect the server.
    CAPTURE_INTERVAL_MS: int = int(os.getenv("CAPTURE_INTERVAL_MS", "1000"))

    # ── File upload limits ────────────────────────────────────────────────────
    # Flask enforces this limit on incoming multipart request bodies.
    # 500 MB accommodates large .pkl model files uploaded by admins.
    MAX_CONTENT_LENGTH: int = 500 * 1024 * 1024  # 500 MB

    # ── Rate limiting (Flask-Limiter) ─────────────────────────────────────────
    # Storage URI for the rate-limit counter.
    # "memory://" is fine for single-process dev; use Redis in production.
    RATELIMIT_STORAGE_URI: str = os.getenv("REDIS_URL", "memory://")

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


# =============================================================================
# ENVIRONMENT-SPECIFIC OVERRIDES
# =============================================================================

class DevelopmentConfig(BaseConfig):
    """
    Local development settings.

    - DEBUG = True enables Flask's interactive debugger and auto-reloader.
    - Session cookie security is relaxed so HTTP (non-HTTPS) localhost works.
    - Log level set to DEBUG for verbose output.
    """
    DEBUG: bool = True
    TESTING: bool = False

    # Allow the session cookie over plain HTTP on localhost
    SESSION_COOKIE_SECURE: bool = False

    LOG_LEVEL: str = "DEBUG"


class TestingConfig(BaseConfig):
    """
    Settings used by pytest.

    - TESTING = True makes Flask propagate exceptions instead of returning 500.
    - WTF_CSRF_ENABLED = False so test clients don't need to supply CSRF tokens.
    - Uses a fixed, predictable SECRET_KEY so token tests are deterministic.
    - Points to a test-specific Supabase project (or can be mocked in tests).
    """
    TESTING: bool = True
    DEBUG: bool = False

    SESSION_COOKIE_SECURE: bool = False
    WTF_CSRF_ENABLED: bool = False

    SECRET_KEY: str = "test-secret-key-not-for-production"

    # Override with a test project URL if you have one; otherwise mock supabase-py
    SUPABASE_URL: str = os.getenv("TEST_SUPABASE_URL", "http://localhost:54321")
    SUPABASE_KEY: str = os.getenv("TEST_SUPABASE_KEY", "test-key")

    LOG_LEVEL: str = "WARNING"   # Quieter output during test runs


class ProductionConfig(BaseConfig):
    """
    Production settings.

    - DEBUG and TESTING are explicitly False.
    - Strict session cookie settings enforced.
    - Relies entirely on environment variables for all secrets.
    """
    DEBUG: bool = False
    TESTING: bool = False

    # Validate required secrets at startup rather than failing at runtime.
    # If SUPABASE_URL or FLASK_SECRET_KEY are missing, startup fails immediately
    # with a clear message rather than a cryptic error during the first request.
    def __init__(self) -> None:
        required = {
            "FLASK_SECRET_KEY":   self.SECRET_KEY,
            "SUPABASE_URL":       self.SUPABASE_URL,
            "SUPABASE_SERVICE_KEY": self.SUPABASE_KEY,
        }
        missing = [k for k, v in required.items() if not v or v.startswith("change-me")]
        if missing:
            raise EnvironmentError(
                f"MoneyTalks: Missing required environment variables: {missing}. "
                "Set them in your .env file or container environment."
            )

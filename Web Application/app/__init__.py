"""
app/__init__.py — MoneyTalks Application Factory
=================================================
The create_app() function is the single entry point for building the Flask
application. Using a factory (rather than a module-level app object) means:

    - Different configs can be passed in for testing vs production.
    - Extensions are cleanly initialised inside the factory scope.
    - Circular imports are avoided (routes.py imports from app, not from each other).

Startup sequence (in order):
    1. Create the Flask app instance.
    2. Load configuration from config.py.
    3. Initialise Flask extensions (Flask-Login, Flask-Limiter, Flask-WTF).
    4. Initialise SupabaseService — this pings Supabase and fails fast if misconfigured.
    5. Download and cache the active .pkl model from Supabase Storage.
    6. Load the .pkl into inference.py's in-memory singleton.
    7. Register Flask Blueprints (routes, auth).
    8. Return the ready-to-serve app.

Called by run.py:
    from app import create_app
    app = create_app()
    app.run()
"""

import logging
import os
import pathlib

from flask import Flask
from flask_login import LoginManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect

from config import get_config
from app.services.supabase_client import init_supabase, SupabaseError
from app.services import inference


# =============================================================================
# FLASK EXTENSIONS
# =============================================================================
# Defined at module level so Blueprints can import them without creating
# circular dependencies. They are bound to the app inside create_app().

login_manager = LoginManager()
limiter       = Limiter(key_func=get_remote_address)
csrf          = CSRFProtect()


# =============================================================================
# APPLICATION FACTORY
# =============================================================================

def create_app(config_override: object | None = None) -> Flask:
    """
    Build, configure, and return the MoneyTalks Flask application.

    Args:
        config_override: Optional config object that replaces the environment-
                         selected config. Used in tests to inject TestingConfig
                         without setting FLASK_ENV:

                             from config import TestingConfig
                             app = create_app(TestingConfig())

    Returns:
        A fully initialised Flask application ready to serve requests.

    Raises:
        SupabaseError:    If Supabase credentials are wrong or unreachable.
        EnvironmentError: If required environment variables are missing
                          (raised by ProductionConfig.__init__).
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # ── 1. Load configuration ─────────────────────────────────────────────────
    cfg = config_override if config_override is not None else get_config()
    app.config.from_object(cfg)

    # ── 2. Configure logging ──────────────────────────────────────────────────
    _configure_logging(app)
    logger = logging.getLogger(__name__)
    logger.info("Starting MoneyTalks | env=%s", os.getenv("FLASK_ENV", "production"))

    # ── 3. Initialise Flask extensions ────────────────────────────────────────
    _init_extensions(app)

    # ── 4. Initialise Supabase (ping-on-startup) ──────────────────────────────
    # This will raise SupabaseError immediately if:
    #   - SUPABASE_URL or SUPABASE_SERVICE_KEY are missing/wrong
    #   - The Supabase project is unreachable (network issue)
    # Failing fast here is intentional — a misconfigured app should not start.
    try:
        init_supabase(app)
        logger.info("Supabase connection established.")
    except SupabaseError as exc:
        logger.critical("Cannot start MoneyTalks: Supabase connection failed. %s", exc)
        raise

    # ── 5. Download active model from Supabase Storage ────────────────────────
    # After init_supabase(), app.supabase is available.
    # We download the active .pkl and write it to the local MODEL_PATH cache.
    _bootstrap_model(app)

    # ── 6. Register Blueprints ────────────────────────────────────────────────
    # Import here (inside the factory) to avoid circular imports.
    from app.routes import main_bp
    from app.auth   import auth_bp

    app.register_blueprint(main_bp)   # Guest detection + API endpoints
    app.register_blueprint(auth_bp, url_prefix="/admin")  # Admin login/dashboard

    logger.info("MoneyTalks is ready.")
    return app


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _configure_logging(app: Flask) -> None:
    """
    Set up structured logging.

    In development, logs go to stdout at DEBUG level.
    In production, the level is controlled by the LOG_LEVEL env var.
    Gunicorn captures stdout so all log output ends up in the container logs.
    """
    log_level = getattr(logging, app.config.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level   = log_level,
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    # Quieten noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _init_extensions(app: Flask) -> None:
    """
    Bind Flask extensions to the app instance.

    Flask-Login:
        - Manages admin sessions. Sets the login view so @login_required
          redirects to /admin/login instead of the default /login.
        - user_loader is defined in auth.py and registered here.

    Flask-Limiter:
        - Rate-limits sensitive endpoints (login, detect).
        - Storage backend is configured via RATELIMIT_STORAGE_URI (memory:// in
          dev, redis:// in production).

    CSRFProtect:
        - Adds CSRF token validation to all state-changing forms.
        - API endpoints that receive JSON/multipart (not HTML forms) are
          exempted per-route with @csrf.exempt in routes.py.
    """
    # Flask-Login
    login_manager.init_app(app)
    login_manager.login_view      = "auth.login"       # redirect target for @login_required
    login_manager.login_message   = "Please sign in to access the admin panel."
    login_manager.session_protection = "strong"        # re-validates IP + user-agent

    # Import the user_loader here to register it with the login_manager.
    # It is defined in auth.py but must be imported after login_manager is bound.
    from app.auth import load_admin_user  # noqa: F401  (registers via @login_manager.user_loader)

    # Flask-Limiter
    limiter.init_app(app)

    # CSRF
    csrf.init_app(app)


def _bootstrap_model(app: Flask) -> None:
    """
    Download the active .pkl model from Supabase Storage and load it into memory.

    Flow:
        1. Ask SupabaseService for the currently deployed model's bytes.
        2. Write the bytes to MODEL_PATH on the local filesystem (cache).
        3. Call inference.load_model(MODEL_PATH) to deserialise and store
           the model in inference.py's thread-safe singleton.

    Why cache locally?
        joblib.load() reads from the filesystem. Storing the file locally means
        the model survives the startup download and can be re-loaded by
        inference.hot_swap_model() without re-downloading from Supabase.

    Why not skip the cache and load from bytes directly?
        joblib can load from a BytesIO buffer, but scikit-learn pipelines with
        external (non-pure-Python) steps (e.g. OpenCV transforms) sometimes
        require a real file path. Caching to disk is the safe default.

    Raises:
        SupabaseError: If the active model cannot be fetched.
        SystemExit:    If no active model is configured (fatal misconfiguration).
    """
    logger = logging.getLogger(__name__)
    model_path = app.config["MODEL_PATH"]

    # Ensure the local directory exists
    pathlib.Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Downloading active model from Supabase Storage...")
        model_bytes = app.supabase.download_active_model_bytes()

        with open(model_path, "wb") as f:
            f.write(model_bytes)
        logger.info("Active model cached at: %s (%d bytes)", model_path, len(model_bytes))

    except SupabaseError as exc:
        logger.critical(
            "Failed to download active model from Supabase Storage. "
            "Ensure at least one ModelVersions row has is_deployed=TRUE "
            "and the model-files bucket contains the corresponding file. "
            "Error: %s", exc
        )
        raise

    # Load the downloaded .pkl into inference.py's in-memory singleton
    inference.load_model(model_path)
    logger.info("ML model loaded into memory from: %s", model_path)

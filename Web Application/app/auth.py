"""
app/auth.py — MoneyTalks Admin Authentication
=============================================
CHANGES FROM PREVIOUS VERSION:
  - login() and login_post() merged into a single view function handling
    both GET and POST. Two separate endpoints on the same URL caused ambiguity
    in url_for() and is non-standard Flask practice.
  - @limiter.limit(..., methods=["POST"]) applies rate limiting only to POST.

No other changes. All Supabase calls, bcrypt logic, and Flask-Login
integration are unchanged from the Data Tier implementation.

Auth flow summary:
  GET  /admin/login     → renders admin/login.html
  POST /admin/login     → validates credentials → session → redirect dashboard
  POST /admin/logout    → destroys session → redirect login
  GET  /admin/dashboard → @login_required → renders dashboard (or → login)
  GET  /admin/images    → @login_required → renders images list
  GET  /admin/models    → @login_required → renders model list
"""

import logging
from datetime import date as date_type

import bcrypt
from flask import (
    Blueprint, flash, redirect, render_template,
    request, url_for,
)
from flask_login import (
    UserMixin, current_user,
    login_required, login_user, logout_user,
)

from app import login_manager, limiter
from app.services.supabase_client import AdminRecord, SupabaseError, get_db

logger = logging.getLogger(__name__)

# url_prefix="/admin" is set when this Blueprint is registered in __init__.py.
auth_bp = Blueprint("auth", __name__, template_folder="templates")


# =============================================================================
# FLASK-LOGIN USER MODEL
# =============================================================================

class AdminUser(UserMixin):
    """
    Flask-Login User model backed by an AdminRecord from Supabase.

    UserMixin provides correct defaults for is_authenticated, is_active,
    is_anonymous. We only override get_id() to return admin_id.
    Password hash is NOT stored here — read once during login and discarded.
    """

    def __init__(self, record: AdminRecord) -> None:
        self.admin_id = record.admin_id
        self.email    = record.email
        self.username = record.username
        self.phone    = record.phone

    def get_id(self) -> str:
        """Return the unique string stored in the session cookie."""
        return self.admin_id


# =============================================================================
# FLASK-LOGIN USER LOADER
# =============================================================================

@login_manager.user_loader
def load_admin_user(admin_id: str):
    """
    Called by Flask-Login on EVERY request to a @login_required route.
    Reconstructs current_user from the session cookie's admin_id.
    Returns None to invalidate the session and redirect to login.
    """
    try:
        record = get_db().get_admin_by_id(admin_id)
        if record is None:
            logger.warning("user_loader: admin_id '%s' not in database.", admin_id)
            return None
        return AdminUser(record)
    except SupabaseError as exc:
        logger.error("user_loader: Supabase error for '%s': %s", admin_id, exc)
        return None


# =============================================================================
# LOGIN / LOGOUT
# =============================================================================

@auth_bp.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute", methods=["POST"])
def login():
    """
    GET  /admin/login — Render the login page.
    POST /admin/login — Validate credentials and create a session.

    GET: redirect to dashboard if already authenticated, else show form.
    POST:
        1. Read email + password from form.
        2. Fetch AdminRecord from Supabase by email.
        3. bcrypt.checkpw() comparison (~100-200ms intentional cost).
        4. On success: login_user() -> redirect to dashboard.
        5. On failure: flash generic error -> re-render login (HTTP 401).

    Security:
        - Rate limited: 5 POST attempts per minute per IP (Flask-Limiter).
        - CSRF token validated by Flask-WTF CSRFProtect before this runs.
        - Generic error message prevents email enumeration.
        - 'next' URL validated before use (prevents open redirects).
    """
    # ── GET ──────────────────────────────────────────────────────────────────
    if request.method == "GET":
        if current_user.is_authenticated:
            return redirect(url_for("auth.dashboard"))
        return render_template("admin/login.html")

    # ── POST ─────────────────────────────────────────────────────────────────
    email    = request.form.get("email",    "").strip().lower()
    password = request.form.get("password", "").strip()

    if not email or not password:
        flash("Please enter both email and password.", "error")
        return render_template("admin/login.html"), 400

    try:
        record = get_db().get_admin_by_email(email)
    except SupabaseError as exc:
        logger.error("login: Supabase error fetching '%s': %s", email, exc)
        flash("A server error occurred. Please try again later.", "error")
        return render_template("admin/login.html"), 500

    # Same branch for "not found" and "wrong password" to prevent email enumeration
    if record is None or not _verify_password(password, record.password):
        logger.warning("Failed login attempt for email: %s (IP: %s)",
                       email, request.remote_addr)
        flash("Invalid email or password.", "error")
        return render_template("admin/login.html"), 401

    # Success
    admin = AdminUser(record)
    login_user(admin, remember=False)
    logger.info("Admin signed in: %s (%s) from %s",
                record.username, record.admin_id, request.remote_addr)

    next_page = request.args.get("next", "")
    if next_page and _is_safe_redirect(next_page):
        return redirect(next_page)
    return redirect(url_for("auth.dashboard"))


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    """
    POST /admin/logout — Destroy the current admin session.

    POST (not GET) prevents browser prefetch and CSRF-based logout attacks.
    The sidebar template includes a CSRF-protected <form> for this button.
    """
    logger.info("Admin signed out: %s (%s)", current_user.username, current_user.admin_id)
    logout_user()
    flash("You have been signed out.", "info")
    return redirect(url_for("auth.login"))


# =============================================================================
# ADMIN PAGE ROUTES
# =============================================================================

@auth_bp.route("/dashboard")
@login_required
def dashboard():
    """
    GET /admin/dashboard — Admin overview page.

    Context injected into admin/dashboard.html:
        admin        — AdminUser (current_user)
        scan_count   — int | "N/A"
        active_model — ModelRecord | None
    """
    try:
        db           = get_db()
        scan_count   = db.get_scan_count()
        active_model = db.get_active_model()
    except SupabaseError as exc:
        logger.error("dashboard: Supabase error: %s", exc)
        scan_count   = "N/A"
        active_model = None

    return render_template(
        "admin/dashboard.html",
        admin        = current_user,
        scan_count   = scan_count,
        active_model = active_model,
    )


@auth_bp.route("/images")
@login_required
def images():
    """
    GET /admin/images — Paginated list of ScannedMoney records.

    Query params: from (YYYY-MM-DD), to (YYYY-MM-DD), page (int, default 1).
    Template: admin/images.html (future sprint).
    """
    def _parse_date(param):
        if not param:
            return None
        try:
            return date_type.fromisoformat(param)
        except ValueError:
            return None

    from_date = _parse_date(request.args.get("from"))
    to_date   = _parse_date(request.args.get("to"))
    page      = max(1, int(request.args.get("page", 1)))

    try:
        records, total = get_db().get_scan_records(
            from_date=from_date, to_date=to_date, page=page, page_size=50,
        )
    except SupabaseError as exc:
        logger.error("images: Supabase error: %s", exc)
        records, total = [], 0

    return render_template(
        "admin/images.html",
        admin=current_user, records=records, total=total,
        page=page, from_date=from_date, to_date=to_date,
    )


@auth_bp.route("/models")
@login_required
def models():
    """
    GET /admin/models — List all model versions.

    Template: admin/models.html (future sprint).
    """
    try:
        model_list = get_db().get_all_models()
    except SupabaseError as exc:
        logger.error("models: Supabase error: %s", exc)
        model_list = []

    return render_template(
        "admin/models.html",
        admin=current_user, models=model_list,
    )


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _verify_password(plain: str, hashed: str) -> bool:
    """bcrypt password comparison. Returns False on any error."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception as exc:
        logger.error("bcrypt.checkpw raised: %s", exc)
        return False


def _is_safe_redirect(url: str) -> bool:
    """
    Accept only relative paths (starts with '/', not '//').
    Prevents open redirect via ?next=https://evil.com.
    """
    return bool(url) and url.startswith("/") and not url.startswith("//")

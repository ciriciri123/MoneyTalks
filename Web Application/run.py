"""
run.py — MoneyTalks Application Entry Point
==========================================
Loads environment variables from .env, then creates and runs the Flask app.

Usage:
    Development:    python run.py
    Production:     gunicorn "run:create_app()" --workers 4 --bind 0.0.0.0:8000

Do NOT use `flask run` — it bypasses the dotenv loading here.
"""

from dotenv import load_dotenv

# Load .env BEFORE importing app (config.py reads env vars at import time)
load_dotenv()

from app import create_app   # noqa: E402  (must come after load_dotenv)

app = create_app()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=app.config.get("DEBUG", False),
    )

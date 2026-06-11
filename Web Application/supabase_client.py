"""
Supabase client — lazy singleton + all DB/storage operations.

All functions raise RuntimeError if SUPABASE_URL / SUPABASE_SERVICE_KEY
are not set, so missing config surfaces clearly at call time, not at import.
"""
import os
import uuid
from datetime import datetime, timezone

from supabase import create_client, Client

_client: Client | None = None

SCAN_BUCKET  = 'scanned-images'
MODEL_BUCKET = 'model-files'


def get_client() -> Client:
    global _client
    if _client is None:
        url = os.environ.get('SUPABASE_URL', '').strip()
        key = os.environ.get('SUPABASE_SERVICE_KEY', '').strip()
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in your .env file."
            )
        _client = create_client(url, key)
    return _client


def is_configured() -> bool:
    return bool(
        os.environ.get('SUPABASE_URL', '').strip() and
        os.environ.get('SUPABASE_SERVICE_KEY', '').strip()
    )


# ── Scans ─────────────────────────────────────────────────────────────────────

def upload_scan(image_bytes: bytes, denomination: str, confidence: float) -> dict:
    db = get_client()
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    image_path = f"{denomination}/{ts}_{uuid.uuid4().hex[:8]}.jpg"

    db.storage.from_(SCAN_BUCKET).upload(
        path=image_path,
        file=image_bytes,
        file_options={"content-type": "image/jpeg"}
    )

    result = db.table('ScannedMoney').insert({
        'denomination': denomination,
        'confidence': round(float(confidence), 4),
        'image_path': image_path,
    }).execute()

    return result.data[0]


def get_scans(from_date: str = None, to_date: str = None,
              limit: int = 50, offset: int = 0) -> list:
    db = get_client()
    query = (db.table('ScannedMoney')
               .select('*')
               .order('scanned_at', desc=True))
    if from_date:
        query = query.gte('scanned_at', from_date)
    if to_date:
        query = query.lte('scanned_at', to_date + 'T23:59:59Z')
    query = query.range(offset, offset + limit - 1)
    return query.execute().data


def get_scan_count() -> int:
    db = get_client()
    result = db.table('ScannedMoney').select('id', count='exact').execute()
    return result.count or 0


def download_scan_image(image_path: str) -> bytes:
    db = get_client()
    return db.storage.from_(SCAN_BUCKET).download(image_path)


# ── Admins ─────────────────────────────────────────────────────────────────────

def get_admin_by_email(email: str) -> dict | None:
    db = get_client()
    result = db.table('Administrator').select('*').eq('email', email).execute()
    return result.data[0] if result.data else None


def get_admin_by_id(admin_id: str) -> dict | None:
    db = get_client()
    result = db.table('Administrator').select('*').eq('id', admin_id).execute()
    return result.data[0] if result.data else None


# ── Models ─────────────────────────────────────────────────────────────────────

def get_model_versions() -> list:
    db = get_client()
    return (db.table('ModelVersions')
              .select('*')
              .order('uploaded_at', desc=True)
              .execute().data)


def get_deployed_model() -> dict | None:
    db = get_client()
    result = db.table('ModelVersions').select('*').eq('is_deployed', True).execute()
    return result.data[0] if result.data else None


def upload_model_zip(file_bytes: bytes, version_string: str, uploaded_by: str) -> dict:
    db = get_client()
    file_path = f"models/v{version_string}_{uuid.uuid4().hex[:8]}.zip"

    db.storage.from_(MODEL_BUCKET).upload(
        path=file_path,
        file=file_bytes,
        file_options={"content-type": "application/zip"}
    )

    result = db.table('ModelVersions').insert({
        'version_string': version_string,
        'file_path': file_path,
        'uploaded_by': uploaded_by,
        'is_deployed': False,
    }).execute()

    return result.data[0]


def deploy_model_version(version_id: str) -> dict:
    db = get_client()
    db.table('ModelVersions').update({'is_deployed': False}).neq('id', version_id).execute()
    result = (db.table('ModelVersions')
                .update({'is_deployed': True})
                .eq('id', version_id)
                .execute())
    return result.data[0]


def download_model_zip(file_path: str) -> bytes:
    db = get_client()
    return db.storage.from_(MODEL_BUCKET).download(file_path)

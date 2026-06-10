"""
app/services/supabase_client.py — MoneyTalks Data Tier Service
==============================================================
This module is the ONLY place in the application that communicates with
Supabase. All Flask routes and other services must go through this module;
nothing else should import supabase-py directly.

Architecture position:
    Browser → Flask Routes (routes.py / auth.py)
                  ↓
            This module (supabase_client.py)
                  ↓
            Supabase Cloud
            ├── PostgreSQL  (tables: ScannedMoney, Administrator, ModelVersions)
            └── Storage     (buckets: scanned-images, model-files)

Design decisions:
    - One module-level SupabaseService singleton initialised at app startup.
    - Every public method returns a typed dataclass or raises a SupabaseError.
    - No raw supabase-py APIResponse objects leak outside this module — callers
      always receive plain Python dicts, lists, bytes, or dataclasses.
    - All storage paths are constructed here; callers pass semantic identifiers
      (image_id, model_id) and this module handles path composition.
    - Errors from the Supabase API are caught, logged, and re-raised as
      SupabaseError so the Flask layer can return clean HTTP error responses.

Dependencies (add to requirements.txt):
    supabase>=2.3.0
    python-dotenv>=1.0.0

Install:
    pip install supabase python-dotenv
"""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional
from uuid import uuid4

from supabase import Client, create_client
from supabase import create_client, Client, ClientOptions
# from supabase.lib.client_options import ClientOptions

logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM EXCEPTION
# =============================================================================

class SupabaseError(Exception):
    """
    Raised whenever a Supabase operation fails.

    Wrapping supabase-py exceptions in our own type means Flask route handlers
    only need to catch SupabaseError — they are insulated from the underlying
    library's exception hierarchy, which may change across supabase-py versions.

    Attributes:
        message: Human-readable description of what went wrong.
        operation: The name of the method that failed (e.g. 'upload_scan_image').
        original: The original exception, if any, for logging purposes.
    """
    def __init__(self, message: str, operation: str = "", original: Exception | None = None):
        super().__init__(message)
        self.operation = operation
        self.original = original

    def __str__(self) -> str:
        base = super().__str__()
        if self.operation:
            return f"[{self.operation}] {base}"
        return base


# =============================================================================
# TYPED RETURN OBJECTS
# =============================================================================
# Using dataclasses instead of raw dicts gives autocomplete in the IDE and
# makes it immediately obvious what shape of data a method returns.

@dataclass
class ScanRecord:
    """Represents one row from the ScannedMoney table."""
    image_id:       int
    image_path:     str
    image_upload:   datetime
    detected_label: str


@dataclass
class AdminRecord:
    """Represents one row from the Administrator table."""
    admin_id:  str
    email:     str
    username:  str
    phone:     Optional[str]
    password:  str           # bcrypt hash — never log or expose this field


@dataclass
class ModelRecord:
    """Represents one row from the ModelVersions table."""
    model_id:     str
    version:      str
    uploaded_by:  str        # admin_id FK
    storage_path: str
    uploaded_at:  datetime
    is_deployed:  bool


# =============================================================================
# SUPABASE SERVICE
# =============================================================================

class SupabaseService:
    """
    Centralised access layer for all Supabase interactions in MoneyTalks.

    Instantiated once by the Flask application factory and stored on the
    app object (app.supabase). Route handlers access it via:

        from flask import current_app
        svc = current_app.supabase

    or via the module-level helper:

        from app.services.supabase_client import get_db

    Bucket name constants are read from Flask app.config so they can be
    overridden per environment without touching this class.
    """

    # ── Storage path templates ─────────────────────────────────────────────────
    # Centralising path composition here means changing the naming convention
    # requires editing only these two lines.
    _SCAN_PATH_TEMPLATE  = "{year}/{month:02d}/{day:02d}/{uuid}.jpg"
    _MODEL_PATH_TEMPLATE = "{model_id}/model.pkl"

    def __init__(self, url: str, key: str, images_bucket: str, models_bucket: str) -> None:
        """
        Create the supabase-py Client and store bucket names.

        Args:
            url:            Supabase project URL (SUPABASE_URL env var).
            key:            Service-role API key (SUPABASE_SERVICE_KEY env var).
                            This key bypasses RLS. It is used ONLY on the server.
            images_bucket:  Name of the scanned-images bucket.
            models_bucket:  Name of the model-files bucket.

        Raises:
            SupabaseError: If the client cannot be created (bad URL / key format).
        """
        if not url or not key:
            raise SupabaseError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must both be set.",
                operation="__init__"
            )

        try:
            # ClientOptions lets us set a custom schema or headers if needed.
            # Using the default public schema here.
            options = ClientOptions(schema="public")
            self._client: Client = create_client(url, key, options)
        except Exception as exc:
            raise SupabaseError(
                f"Failed to create Supabase client: {exc}",
                operation="__init__",
                original=exc,
            ) from exc

        self._images_bucket = images_bucket
        self._models_bucket = models_bucket
        logger.info("SupabaseService initialised. images_bucket=%s models_bucket=%s",
                    images_bucket, models_bucket)

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _table(self, name: str):
        """
        Return a supabase-py QueryBuilder for the given table name.

        Wrapping self._client.table() in a helper method means we can add
        uniform logging or tracing here later without touching every call site.
        """
        return self._client.table(name)

    def _storage_bucket(self, bucket: str):
        """Return the supabase-py StorageClient for a given bucket name."""
        return self._client.storage.from_(bucket)

    @staticmethod
    def _build_scan_path(unique_id: str, upload_dt: datetime) -> str:
        """
        Compose the Supabase Storage path for a new scanned image.

        Example output: scans/2026/03/28/f47ac10b-58cc-4372-a567.jpg

        Organising by date makes it cheap to query (prefix-match) or delete
        (by date prefix) directly in Supabase Storage if needed.
        """
        return SupabaseService._SCAN_PATH_TEMPLATE.format(
            year=upload_dt.year,
            month=upload_dt.month,
            day=upload_dt.day,
            uuid=unique_id,
        )

    @staticmethod
    def _build_model_path(model_id: str) -> str:
        """
        Compose the Supabase Storage path for a model .pkl file.

        Example output: models/MDL-001/model.pkl
        """
        return SupabaseService._MODEL_PATH_TEMPLATE.format(model_id=model_id)

    @staticmethod
    def _parse_datetime(value: str | datetime | None) -> datetime:
        """
        Normalise a datetime value returned by supabase-py to a timezone-aware
        Python datetime object in UTC.

        Supabase returns ISO-8601 strings like '2026-03-28T10:15:30.123456+00:00'.
        """
        if value is None:
            return datetime.now(tz=timezone.utc)
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        # Parse the ISO string
        try:
            dt = datetime.fromisoformat(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            logger.warning("Could not parse datetime value: %r — using UTC now", value)
            return datetime.now(tz=timezone.utc)

    # =========================================================================
    # SCAN IMAGE OPERATIONS
    # =========================================================================

    def upload_scan_image(
        self,
        image_bytes: bytes,
        detected_label: str,
    ) -> ScanRecord:
        """
        Upload a JPEG frame to Supabase Storage and insert a ScannedMoney record.

        This method performs TWO operations in sequence:
          1. Upload the image file to the scanned-images bucket.
          2. Insert a row into ScannedMoney with the storage path and label.

        If step 1 fails, step 2 is never attempted (no orphaned DB records).
        If step 2 fails after step 1 succeeds, the orphaned file is logged
        but NOT deleted automatically (it can be cleaned up by an admin).

        Args:
            image_bytes:     Raw JPEG bytes captured from the browser canvas.
            detected_label:  Denomination label from the ML model, e.g. 'Rp 50.000'.

        Returns:
            ScanRecord dataclass with all fields populated, including the
            auto-assigned image_id from the database.

        Raises:
            SupabaseError: On any storage or database failure.
        """
        upload_dt  = datetime.now(tz=timezone.utc)
        unique_id  = str(uuid4())
        image_path = self._build_scan_path(unique_id, upload_dt)

        # ── Step 1: Upload image to Supabase Storage ──────────────────────────
        try:
            self._storage_bucket(self._images_bucket).upload(
                path=image_path,
                file=image_bytes,
                file_options={"content-type": "image/jpeg", "upsert": "false"},
            )
            logger.debug("Image uploaded to storage: %s", image_path)
        except Exception as exc:
            raise SupabaseError(
                f"Storage upload failed for path '{image_path}': {exc}",
                operation="upload_scan_image",
                original=exc,
            ) from exc

        # ── Step 2: Insert metadata row into ScannedMoney ─────────────────────
        try:
            response = (
                self._table("ScannedMoney")
                .insert({
                    "image_path":     image_path,
                    "image_upload":   upload_dt.isoformat(),
                    "detected_label": detected_label,
                })
                .execute()
            )
            # supabase-py returns the inserted row(s) in response.data
            row = response.data[0]
            logger.info("ScannedMoney record created: image_id=%s label=%s",
                        row["image_id"], detected_label)
            return ScanRecord(
                image_id       = row["image_id"],
                image_path     = row["image_path"],
                image_upload   = self._parse_datetime(row["image_upload"]),
                detected_label = row["detected_label"],
            )
        except SupabaseError:
            raise
        except Exception as exc:
            # The file was uploaded but the DB record failed.
            # Log the orphaned path so it can be manually reconciled.
            logger.error(
                "DB insert failed after successful storage upload. "
                "Orphaned file: %s. Error: %s",
                image_path, exc
            )
            raise SupabaseError(
                f"Database insert failed after storage upload: {exc}",
                operation="upload_scan_image",
                original=exc,
            ) from exc

    def get_scan_records(
        self,
        from_date: Optional[date] = None,
        to_date:   Optional[date] = None,
        page:      int = 1,
        page_size: int = 50,
    ) -> tuple[list[ScanRecord], int]:
        """
        Retrieve a paginated, optionally date-filtered list of scan records.

        Used by the admin /images page to browse and select records for download.
        Returns records newest-first (ORDER BY image_upload DESC).

        Args:
            from_date:  Include only records with image_upload >= this date (UTC).
            to_date:    Include only records with image_upload <= this date (UTC), inclusive.
            page:       1-based page number.
            page_size:  Number of records per page (max 200 enforced below).

        Returns:
            Tuple of (list of ScanRecord, total_count).
            total_count allows the admin UI to render pagination controls.

        Raises:
            SupabaseError: On database query failure.
        """
        page_size = min(page_size, 200)  # hard cap to protect memory
        offset    = (page - 1) * page_size

        try:
            query = (
                self._table("ScannedMoney")
                .select("*", count="exact")    # count="exact" populates response.count
                .order("image_upload", desc=True)
                .range(offset, offset + page_size - 1)
            )
            # Supabase date filter: convert Python date to ISO-8601 strings
            if from_date:
                query = query.gte("image_upload", from_date.isoformat())
            if to_date:
                # Add one day so 'to_date' is inclusive (< next day at midnight)
                to_dt = datetime(to_date.year, to_date.month, to_date.day,
                                 23, 59, 59, tzinfo=timezone.utc)
                query = query.lte("image_upload", to_dt.isoformat())

            response = query.execute()
            records  = [
                ScanRecord(
                    image_id       = r["image_id"],
                    image_path     = r["image_path"],
                    image_upload   = self._parse_datetime(r["image_upload"]),
                    detected_label = r["detected_label"],
                )
                for r in response.data
            ]
            total = response.count or 0
            return records, total

        except Exception as exc:
            raise SupabaseError(
                f"Failed to fetch scan records: {exc}",
                operation="get_scan_records",
                original=exc,
            ) from exc

    def download_scans_as_zip(
        self,
        from_date: Optional[date] = None,
        to_date:   Optional[date] = None,
    ) -> bytes:
        """
        Build and return an in-memory ZIP archive of scanned images.

        Fetches all matching image_path values from the database (no pagination —
        this is an admin bulk-export operation), then downloads each file from
        Supabase Storage and writes it into a BytesIO zip buffer.

        The zip archive is streamed back to the browser via Flask's
        send_file() with as_attachment=True.

        Design note: this streams directly from Supabase Storage into the zip
        buffer without writing anything to the server's local filesystem.

        Args:
            from_date:  Start of date range (inclusive). None = no lower bound.
            to_date:    End of date range (inclusive). None = no upper bound.

        Returns:
            Raw bytes of the .zip archive.

        Raises:
            SupabaseError: If the database query or any storage download fails.
        """
        # Fetch all matching records (no page limit — admin download)
        records, total = self.get_scan_records(
            from_date=from_date,
            to_date=to_date,
            page=1,
            page_size=10_000,  # practical upper bound; adjust if needed
        )
        logger.info("Building zip for %d scan images (from=%s to=%s)", total, from_date, to_date)

        zip_buffer = io.BytesIO()
        bucket     = self._storage_bucket(self._images_bucket)

        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for record in records:
                try:
                    file_bytes = bucket.download(record.image_path)
                    # Use just the last segment of the path as the filename inside the zip
                    zip_filename = record.image_path.split("/")[-1]
                    zf.writestr(zip_filename, file_bytes)
                except Exception as exc:
                    # Log and skip missing files rather than aborting the whole zip
                    logger.warning(
                        "Could not download %s for zip archive: %s",
                        record.image_path, exc
                    )

        zip_buffer.seek(0)
        return zip_buffer.read()

    def get_scan_count(self) -> int:
        """
        Return the total number of rows in ScannedMoney.
        Used by the admin dashboard summary card.
        """
        try:
            response = self._table("ScannedMoney").select("image_id", count="exact").execute()
            return response.count or 0
        except Exception as exc:
            raise SupabaseError(
                f"Failed to count scan records: {exc}",
                operation="get_scan_count",
                original=exc,
            ) from exc

    # =========================================================================
    # ADMINISTRATOR OPERATIONS
    # =========================================================================

    def get_admin_by_email(self, email: str) -> Optional[AdminRecord]:
        """
        Look up an administrator record by email address.

        Called by auth.py during the login flow. Returns None if no matching
        row is found (caller is responsible for returning a 401/redirect).

        Args:
            email: The email address submitted in the login form.

        Returns:
            AdminRecord if found, None if not found.

        Raises:
            SupabaseError: On database query failure (distinct from "not found").
        """
        try:
            response = (
                self._table("Administrator")
                .select("*")
                .eq("email", email)
                .limit(1)
                .execute()
            )
            if not response.data:
                logger.debug("No admin found for email: %s", email)
                return None

            row = response.data[0]
            return AdminRecord(
                admin_id = row["admin_id"],
                email    = row["email"],
                username = row["username"],
                phone    = row.get("phone"),
                password = row["password"],    # bcrypt hash
            )
        except Exception as exc:
            raise SupabaseError(
                f"Failed to fetch admin by email: {exc}",
                operation="get_admin_by_email",
                original=exc,
            ) from exc

    def get_admin_by_id(self, admin_id: str) -> Optional[AdminRecord]:
        """
        Look up an administrator record by primary key.

        Called by Flask-Login's user_loader callback on every authenticated
        request to reconstruct the user object from the session cookie.

        Args:
            admin_id: The admin_id value stored in the Flask session.

        Returns:
            AdminRecord if found, None if not found (triggers session invalidation).

        Raises:
            SupabaseError: On database query failure.
        """
        try:
            response = (
                self._table("Administrator")
                .select("*")
                .eq("admin_id", admin_id)
                .limit(1)
                .execute()
            )
            if not response.data:
                return None

            row = response.data[0]
            return AdminRecord(
                admin_id = row["admin_id"],
                email    = row["email"],
                username = row["username"],
                phone    = row.get("phone"),
                password = row["password"],
            )
        except Exception as exc:
            raise SupabaseError(
                f"Failed to fetch admin by id '{admin_id}': {exc}",
                operation="get_admin_by_id",
                original=exc,
            ) from exc

    # =========================================================================
    # MODEL VERSION OPERATIONS
    # =========================================================================

    def get_all_models(self) -> list[ModelRecord]:
        """
        Return all rows from ModelVersions, newest-first.

        Used by the admin /models page to list all uploaded model versions
        and show which one is currently deployed.

        Returns:
            List of ModelRecord, ordered by uploaded_at descending.

        Raises:
            SupabaseError: On database query failure.
        """
        try:
            response = (
                self._table("ModelVersions")
                .select("*")
                .order("uploaded_at", desc=True)
                .execute()
            )
            return [
                ModelRecord(
                    model_id     = r["model_id"],
                    version      = r["version"],
                    uploaded_by  = r["uploaded_by"],
                    storage_path = r["storage_path"],
                    uploaded_at  = self._parse_datetime(r["uploaded_at"]),
                    is_deployed  = r["is_deployed"],
                )
                for r in response.data
            ]
        except Exception as exc:
            raise SupabaseError(
                f"Failed to fetch model versions: {exc}",
                operation="get_all_models",
                original=exc,
            ) from exc

    def get_active_model(self) -> Optional[ModelRecord]:
        """
        Return the single row where is_deployed = TRUE.

        Called at application startup (app/__init__.py) to identify which
        .pkl file to download and load into memory.

        Returns:
            ModelRecord of the active model, or None if no model is deployed.
            (None should be treated as a fatal configuration error at startup.)

        Raises:
            SupabaseError: On database query failure.
        """
        try:
            response = (
                self._table("ModelVersions")
                .select("*")
                .eq("is_deployed", True)
                .limit(1)
                .execute()
            )
            if not response.data:
                logger.warning("No active (is_deployed=TRUE) model found in ModelVersions.")
                return None

            r = response.data[0]
            return ModelRecord(
                model_id     = r["model_id"],
                version      = r["version"],
                uploaded_by  = r["uploaded_by"],
                storage_path = r["storage_path"],
                uploaded_at  = self._parse_datetime(r["uploaded_at"]),
                is_deployed  = r["is_deployed"],
            )
        except Exception as exc:
            raise SupabaseError(
                f"Failed to fetch active model: {exc}",
                operation="get_active_model",
                original=exc,
            ) from exc

    def upload_model_file(
        self,
        model_id:    str,
        version:     str,
        admin_id:    str,
        pkl_bytes:   bytes,
    ) -> ModelRecord:
        """
        Upload a new .pkl file to Supabase Storage and create a ModelVersions record.

        The new model is NOT deployed (is_deployed = FALSE).
        The admin must separately call deploy_model() to activate it.

        Args:
            model_id:   Unique identifier for this model version, e.g. 'MDL-002'.
                        Caller (route handler) is responsible for generating this.
            version:    Semantic version string, e.g. '2.1.0'.
            admin_id:   admin_id of the administrator uploading the file.
            pkl_bytes:  Raw bytes of the .pkl file from the multipart upload.

        Returns:
            ModelRecord with is_deployed = False.

        Raises:
            SupabaseError: On storage upload or database insert failure.
        """
        storage_path = self._build_model_path(model_id)

        # ── Step 1: Upload .pkl to Supabase Storage ───────────────────────────
        try:
            self._storage_bucket(self._models_bucket).upload(
                path=storage_path,
                file=pkl_bytes,
                file_options={"content-type": "application/octet-stream", "upsert": "false"},
            )
            logger.info("Model file uploaded: %s", storage_path)
        except Exception as exc:
            raise SupabaseError(
                f"Storage upload failed for model '{model_id}': {exc}",
                operation="upload_model_file",
                original=exc,
            ) from exc

        # ── Step 2: Insert ModelVersions record ───────────────────────────────
        try:
            now      = datetime.now(tz=timezone.utc)
            response = (
                self._table("ModelVersions")
                .insert({
                    "model_id":     model_id,
                    "version":      version,
                    "uploaded_by":  admin_id,
                    "storage_path": storage_path,
                    "uploaded_at":  now.isoformat(),
                    "is_deployed":  False,
                })
                .execute()
            )
            row = response.data[0]
            logger.info("ModelVersions record created: model_id=%s version=%s", model_id, version)
            return ModelRecord(
                model_id     = row["model_id"],
                version      = row["version"],
                uploaded_by  = row["uploaded_by"],
                storage_path = row["storage_path"],
                uploaded_at  = self._parse_datetime(row["uploaded_at"]),
                is_deployed  = row["is_deployed"],
            )
        except Exception as exc:
            logger.error("DB insert failed after model storage upload. Orphaned: %s", storage_path)
            raise SupabaseError(
                f"Database insert failed after model storage upload: {exc}",
                operation="upload_model_file",
                original=exc,
            ) from exc

    def deploy_model(self, model_id: str) -> ModelRecord:
        """
        Set is_deployed = TRUE for the given model_id.

        The database trigger enforce_single_deployed_model() automatically
        sets is_deployed = FALSE on all other rows, ensuring exactly one
        model is active at any time. The trigger runs inside the same
        database transaction as this UPDATE, so it is atomic.

        IMPORTANT: This method only updates the database flag.
        The calling route handler (routes.py) is responsible for:
          1. Calling download_active_model_bytes() to get the new .pkl bytes.
          2. Writing them to the local MODEL_PATH cache.
          3. Calling inference.hot_swap_model() to reload the in-memory model.

        Args:
            model_id: The model_id to activate.

        Returns:
            Updated ModelRecord with is_deployed = True.

        Raises:
            SupabaseError: If the model_id does not exist or the update fails.
        """
        try:
            response = (
                self._table("ModelVersions")
                .update({"is_deployed": True})
                .eq("model_id", model_id)
                .execute()
            )
            if not response.data:
                raise SupabaseError(
                    f"model_id '{model_id}' not found in ModelVersions.",
                    operation="deploy_model",
                )

            row = response.data[0]
            logger.info("Model deployed: model_id=%s version=%s", row["model_id"], row["version"])
            return ModelRecord(
                model_id     = row["model_id"],
                version      = row["version"],
                uploaded_by  = row["uploaded_by"],
                storage_path = row["storage_path"],
                uploaded_at  = self._parse_datetime(row["uploaded_at"]),
                is_deployed  = row["is_deployed"],
            )
        except SupabaseError:
            raise
        except Exception as exc:
            raise SupabaseError(
                f"Failed to deploy model '{model_id}': {exc}",
                operation="deploy_model",
                original=exc,
            ) from exc

    def download_active_model_bytes(self) -> bytes:
        """
        Download the .pkl file for the currently deployed model and return its bytes.

        Called in two scenarios:
          1. Application startup — to populate the local MODEL_PATH cache.
          2. After deploy_model() — to refresh the local cache with the new model.

        The bytes are written to the local filesystem by the caller (app/__init__.py
        or the deploy route handler), not by this method. This keeps I/O concerns
        in the application layer and storage concerns in this service.

        Returns:
            Raw bytes of the active model's .pkl file.

        Raises:
            SupabaseError: If no active model exists or the storage download fails.
        """
        active = self.get_active_model()
        if active is None:
            raise SupabaseError(
                "Cannot download model: no model is marked as is_deployed=TRUE.",
                operation="download_active_model_bytes",
            )

        try:
            file_bytes = self._storage_bucket(self._models_bucket).download(active.storage_path)
            logger.info("Active model downloaded: %s (%d bytes)", active.storage_path, len(file_bytes))
            return file_bytes
        except Exception as exc:
            raise SupabaseError(
                f"Failed to download model file '{active.storage_path}': {exc}",
                operation="download_active_model_bytes",
                original=exc,
            ) from exc

    def download_model_bytes_by_id(self, model_id: str) -> bytes:
        """
        Download the .pkl bytes for a specific model version by model_id.

        Used when deploying a specific model — fetch the file after the DB
        flag has been updated.

        Args:
            model_id: The model_id of the version to download.

        Returns:
            Raw bytes of the .pkl file.

        Raises:
            SupabaseError: If the model_id is not found or download fails.
        """
        storage_path = self._build_model_path(model_id)
        try:
            file_bytes = self._storage_bucket(self._models_bucket).download(storage_path)
            logger.debug("Downloaded model %s (%d bytes)", storage_path, len(file_bytes))
            return file_bytes
        except Exception as exc:
            raise SupabaseError(
                f"Failed to download model '{model_id}' from storage: {exc}",
                operation="download_model_bytes_by_id",
                original=exc,
            ) from exc

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def ping(self) -> bool:
        """
        Test the Supabase connection by making a minimal DB query.

        Called at application startup to fail fast if the credentials are wrong
        or the Supabase project is unreachable.

        Returns:
            True if the connection is healthy.

        Raises:
            SupabaseError: If the ping query fails.
        """
        try:
            # A SELECT on a known table with LIMIT 0 returns instantly and
            # costs essentially nothing, but proves the connection works.
            self._table("Administrator").select("admin_id").limit(0).execute()
            logger.info("Supabase connection: OK")
            return True
        except Exception as exc:
            raise SupabaseError(
                f"Supabase health check failed: {exc}",
                operation="ping",
                original=exc,
            ) from exc


# =============================================================================
# MODULE-LEVEL SINGLETON + ACCESSOR
# =============================================================================
# The singleton is set by the application factory (app/__init__.py) after Flask
# reads config. It must not be instantiated at import time because config values
# (SUPABASE_URL, SUPABASE_KEY) are not available until the app is created.

_service: Optional[SupabaseService] = None


def init_supabase(app) -> SupabaseService:
    """
    Initialise the module-level SupabaseService singleton from Flask app config.

    Called ONCE inside the application factory (app/__init__.py):

        from app.services.supabase_client import init_supabase
        init_supabase(app)

    Args:
        app: The Flask application instance (after app.config is populated).

    Returns:
        The initialised SupabaseService singleton.

    Raises:
        SupabaseError: If the client cannot be created or the ping fails.
    """
    global _service

    svc = SupabaseService(
        url            = app.config["SUPABASE_URL"],
        key            = app.config["SUPABASE_KEY"],
        images_bucket  = app.config["SUPABASE_IMAGES_BUCKET"],
        models_bucket  = app.config["SUPABASE_MODELS_BUCKET"],
    )
    svc.ping()        # fail fast if credentials are wrong
    _service = svc

    # Also store it on the app object so it can be accessed via current_app.supabase
    app.supabase = svc
    return svc


def get_db() -> SupabaseService:
    """
    Return the initialised SupabaseService singleton.

    Convenience accessor for use in route handlers and other services:

        from app.services.supabase_client import get_db
        db = get_db()
        record = db.upload_scan_image(image_bytes, label)

    Raises:
        RuntimeError: If called before init_supabase() (i.e. outside app context).
    """
    if _service is None:
        raise RuntimeError(
            "SupabaseService has not been initialised. "
            "Ensure init_supabase(app) is called inside the application factory "
            "before any route handler runs."
        )
    return _service

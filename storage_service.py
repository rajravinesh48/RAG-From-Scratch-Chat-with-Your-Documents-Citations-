"""
Persistent Supabase Storage helper for the Scholar RAG project.

Required environment variables:
    SUPABASE_URL=https://your-project.supabase.co
    SUPABASE_SERVICE_KEY=your-service-role-key
    SUPABASE_BUCKET=rag-documents

Optional environment variables:
    SUPABASE_FOLDER=documents
    DOCUMENTS_DIR=documents
    ALLOWED_DOCUMENT_EXTENSIONS=.pdf,.docx,.txt

Important:
    SUPABASE_SERVICE_KEY is a server-side secret. Never expose it in HTML,
    JavaScript, GitHub, or a public repository.
"""

from __future__ import annotations

import mimetypes
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client


load_dotenv(override=True)


SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "rag-documents").strip()
SUPABASE_FOLDER = os.getenv("SUPABASE_FOLDER", "documents").strip().strip("/")
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", "documents"))

ALLOWED_DOCUMENT_EXTENSIONS = {
    value if value.startswith(".") else f".{value}"
    for value in (
        item.strip().lower()
        for item in os.getenv(
            "ALLOWED_DOCUMENT_EXTENSIONS",
            ".pdf,.docx,.txt",
        ).split(",")
    )
    if value
}


class StorageServiceError(RuntimeError):
    """Raised when a Supabase Storage operation fails."""


def is_storage_configured() -> bool:
    """Returns True when all required Supabase settings are present."""
    return bool(SUPABASE_URL and SUPABASE_SERVICE_KEY and SUPABASE_BUCKET)


def validate_storage_configuration() -> None:
    """Raises a readable error when configuration is incomplete."""
    missing = []

    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_SERVICE_KEY:
        missing.append("SUPABASE_SERVICE_KEY")
    if not SUPABASE_BUCKET:
        missing.append("SUPABASE_BUCKET")

    if missing:
        raise StorageServiceError(
            "Supabase Storage is not configured. Missing: "
            + ", ".join(missing)
        )


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Creates and caches the server-side Supabase client."""
    validate_storage_configuration()

    try:
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception as exc:
        raise StorageServiceError(
            f"Unable to create Supabase client: {exc}"
        ) from exc


def _safe_filename(filename: str) -> str:
    """Sanitizes a filename and validates its extension."""
    original_name = Path(str(filename or "").replace("\\", "/")).name.strip()

    if not original_name:
        raise StorageServiceError("A valid filename is required.")

    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_DOCUMENT_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))
        raise StorageServiceError(
            f"Unsupported file type '{extension or 'none'}'. "
            f"Allowed types: {allowed}."
        )

    safe_stem = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        Path(original_name).stem,
    ).strip("._-")

    if not safe_stem:
        safe_stem = "document"

    return f"{safe_stem}{extension}"


def _remote_path(filename: str) -> str:
    """Builds the object path used inside the Supabase bucket."""
    safe_name = _safe_filename(filename)
    return f"{SUPABASE_FOLDER}/{safe_name}" if SUPABASE_FOLDER else safe_name


def _response_value(item: Any, key: str, default: Any = None) -> Any:
    """Reads Supabase list values returned as dictionaries or objects."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _read_upload_bytes(file_source: Any) -> bytes:
    """Reads upload data from FileStorage, path, bytes, or file object."""
    if isinstance(file_source, bytes):
        return file_source

    if isinstance(file_source, bytearray):
        return bytes(file_source)

    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        if not path.is_file():
            raise StorageServiceError(f"Upload file does not exist: {path}")
        return path.read_bytes()

    stream = getattr(file_source, "stream", None)
    if stream is not None:
        try:
            stream.seek(0)
        except Exception:
            pass

        data = stream.read()

        try:
            stream.seek(0)
        except Exception:
            pass

        return bytes(data)

    read_method = getattr(file_source, "read", None)
    if callable(read_method):
        try:
            file_source.seek(0)
        except Exception:
            pass

        data = read_method()

        try:
            file_source.seek(0)
        except Exception:
            pass

        if isinstance(data, str):
            return data.encode("utf-8")
        return bytes(data)

    raise StorageServiceError(
        "Unsupported upload source. Pass a Flask FileStorage object, "
        "a file path, bytes, or a binary file object."
    )


def upload_document(
    file_source: Any,
    filename: str | None = None,
    *,
    upsert: bool = True,
    content_type: str | None = None,
) -> str:
    """Uploads one PDF, DOCX, or TXT document to Supabase Storage."""
    detected_name = (
        filename
        or getattr(file_source, "filename", None)
        or (
            Path(file_source).name
            if isinstance(file_source, (str, Path))
            else None
        )
    )

    if not detected_name:
        raise StorageServiceError("Filename could not be detected.")

    remote_path = _remote_path(detected_name)
    data = _read_upload_bytes(file_source)

    if not data:
        raise StorageServiceError("The uploaded document is empty.")

    mime_type = (
        content_type
        or mimetypes.guess_type(remote_path)[0]
        or "application/octet-stream"
    )

    file_options = {
        "cache-control": "3600",
        "content-type": mime_type,
        "upsert": "true" if upsert else "false",
    }

    try:
        (
            get_supabase_client()
            .storage
            .from_(SUPABASE_BUCKET)
            .upload(
                path=remote_path,
                file=data,
                file_options=file_options,
            )
        )
    except Exception as exc:
        raise StorageServiceError(
            f"Unable to upload '{remote_path}' to Supabase Storage: {exc}"
        ) from exc

    return remote_path


def list_documents() -> list[str]:
    """Lists supported document filenames in the configured bucket folder."""
    try:
        response = (
            get_supabase_client()
            .storage
            .from_(SUPABASE_BUCKET)
            .list(
                SUPABASE_FOLDER,
                {
                    "limit": 1000,
                    "offset": 0,
                    "sortBy": {
                        "column": "name",
                        "order": "asc",
                    },
                },
            )
        )
    except Exception as exc:
        raise StorageServiceError(
            f"Unable to list Supabase documents: {exc}"
        ) from exc

    documents = []

    for item in response or []:
        name = str(_response_value(item, "name", "") or "").strip()

        if name and Path(name).suffix.lower() in ALLOWED_DOCUMENT_EXTENSIONS:
            documents.append(name)

    return sorted(set(documents), key=str.lower)


def download_document(
    filename: str,
    destination_dir: str | Path = DOCUMENTS_DIR,
) -> Path:
    """Downloads one private Supabase document to a local directory."""
    safe_name = _safe_filename(filename)
    remote_path = _remote_path(safe_name)

    try:
        data = (
            get_supabase_client()
            .storage
            .from_(SUPABASE_BUCKET)
            .download(remote_path)
        )
    except Exception as exc:
        raise StorageServiceError(
            f"Unable to download '{remote_path}': {exc}"
        ) from exc

    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    local_path = destination / safe_name

    try:
        local_path.write_bytes(bytes(data))
    except Exception as exc:
        raise StorageServiceError(
            f"Unable to save downloaded file '{local_path}': {exc}"
        ) from exc

    return local_path


def download_all_documents(
    destination_dir: str | Path = DOCUMENTS_DIR,
    *,
    clear_local: bool = False,
) -> list[Path]:
    """Downloads every supported Supabase document to the local folder."""
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)

    if clear_local:
        for local_file in destination.iterdir():
            if (
                local_file.is_file()
                and local_file.suffix.lower() in ALLOWED_DOCUMENT_EXTENSIONS
            ):
                local_file.unlink()

    downloaded = []

    for filename in list_documents():
        downloaded.append(download_document(filename, destination))

    return downloaded


def sync_documents_to_local(
    destination_dir: str | Path = DOCUMENTS_DIR,
) -> list[Path]:
    """
    Makes the local documents directory match Supabase Storage.

    Call this immediately before ingest_documents() on Render.
    """
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    remote_names = set(list_documents())

    for local_file in destination.iterdir():
        if (
            local_file.is_file()
            and local_file.suffix.lower() in ALLOWED_DOCUMENT_EXTENSIONS
            and local_file.name not in remote_names
        ):
            local_file.unlink()

    downloaded = []

    for filename in sorted(remote_names, key=str.lower):
        downloaded.append(download_document(filename, destination))

    return downloaded


def delete_document(filename: str) -> str:
    """Deletes one document from Supabase Storage."""
    remote_path = _remote_path(filename)

    try:
        (
            get_supabase_client()
            .storage
            .from_(SUPABASE_BUCKET)
            .remove([remote_path])
        )
    except Exception as exc:
        raise StorageServiceError(
            f"Unable to delete '{remote_path}' from Supabase Storage: {exc}"
        ) from exc

    return remote_path


def document_exists(filename: str) -> bool:
    """Returns True when a filename exists in the configured bucket folder."""
    return _safe_filename(filename) in set(list_documents())


def get_storage_status() -> dict[str, Any]:
    """Returns a small status dictionary for diagnostics."""
    status: dict[str, Any] = {
        "configured": is_storage_configured(),
        "bucket": SUPABASE_BUCKET,
        "folder": SUPABASE_FOLDER,
        "document_count": 0,
        "error": None,
    }

    if not status["configured"]:
        status["error"] = "Supabase environment variables are incomplete."
        return status

    try:
        status["document_count"] = len(list_documents())
    except StorageServiceError as exc:
        status["error"] = str(exc)

    return status


if __name__ == "__main__":
    print("Supabase Storage status:")

    for key, value in get_storage_status().items():
        print(f"  {key}: {value}")

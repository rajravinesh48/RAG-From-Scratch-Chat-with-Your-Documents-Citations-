"""
GitHub-backed document storage for a small Flask/RAG demo.

Uploaded PDF, DOCX, and TXT files are committed directly to the repository's
documents folder through GitHub's REST Contents API.

Required environment variables:
    GITHUB_TOKEN=github_pat_...
    GITHUB_OWNER=your-github-username
    GITHUB_REPO=your-repository-name

Optional environment variables:
    GITHUB_BRANCH=main
    GITHUB_DOCUMENTS_PATH=documents
    GITHUB_API_VERSION=2026-03-10
    DOCUMENTS_DIR=documents
    ALLOWED_DOCUMENT_EXTENSIONS=.pdf,.docx,.txt
    GITHUB_MAX_UPLOAD_BYTES=26214400

Security:
    Use a fine-grained GitHub token restricted to only this repository, with
    Repository permissions -> Contents -> Read and write.

Important deployment behavior:
    Every upload, replacement, or deletion creates a Git commit. When Render
    auto-deploy is enabled for the branch, that commit can trigger a redeploy.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv


load_dotenv()


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_OWNER = os.getenv("GITHUB_OWNER", "").strip()
GITHUB_REPO = os.getenv("GITHUB_REPO", "").strip()
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main").strip() or "main"
GITHUB_DOCUMENTS_PATH = (
    os.getenv("GITHUB_DOCUMENTS_PATH", "documents").strip().strip("/")
)
GITHUB_API_VERSION = (
    os.getenv("GITHUB_API_VERSION", "2026-03-10").strip()
    or "2026-03-10"
)

DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", "documents"))

try:
    GITHUB_MAX_UPLOAD_BYTES = int(
        os.getenv("GITHUB_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024))
    )
except ValueError:
    GITHUB_MAX_UPLOAD_BYTES = 25 * 1024 * 1024

ALLOWED_DOCUMENT_EXTENSIONS = {
    value.strip().lower()
    if value.strip().startswith(".")
    else f".{value.strip().lower()}"
    for value in os.getenv(
        "ALLOWED_DOCUMENT_EXTENSIONS",
        ".pdf,.docx,.txt",
    ).split(",")
    if value.strip()
}

API_ROOT = "https://api.github.com"


class GitHubStorageError(RuntimeError):
    """Raised when a GitHub storage operation fails."""


def is_storage_configured() -> bool:
    """Return True when all required GitHub settings are present."""
    return bool(GITHUB_TOKEN and GITHUB_OWNER and GITHUB_REPO)


def validate_storage_configuration() -> None:
    """Raise an understandable error when required settings are missing."""
    missing: list[str] = []

    if not GITHUB_TOKEN:
        missing.append("GITHUB_TOKEN")
    if not GITHUB_OWNER:
        missing.append("GITHUB_OWNER")
    if not GITHUB_REPO:
        missing.append("GITHUB_REPO")

    if missing:
        raise GitHubStorageError(
            "GitHub storage is not configured. Missing environment "
            f"variable(s): {', '.join(missing)}."
        )


def _safe_filename(filename: str) -> str:
    """Create a safe document filename and block path traversal."""
    original = Path(str(filename or "").replace("\\", "/")).name.strip()

    if not original:
        raise GitHubStorageError("A valid filename is required.")

    extension = Path(original).suffix.lower()

    if extension not in ALLOWED_DOCUMENT_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))
        raise GitHubStorageError(
            f"Unsupported file type '{extension or 'none'}'. "
            f"Allowed types: {allowed}."
        )

    safe_stem = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        Path(original).stem,
    ).strip("._-")

    if not safe_stem:
        safe_stem = "document"

    return f"{safe_stem}{extension}"


def _repository_path(filename: str) -> str:
    """Return the complete path of a document inside the repository."""
    safe_name = _safe_filename(filename)

    if GITHUB_DOCUMENTS_PATH:
        return f"{GITHUB_DOCUMENTS_PATH}/{safe_name}"

    return safe_name


def _contents_url(path: str = "", *, ref: str | None = None) -> str:
    """Build a GitHub Contents API URL."""
    encoded_path = quote(path.strip("/"), safe="/")

    url = (
        f"{API_ROOT}/repos/"
        f"{quote(GITHUB_OWNER, safe='')}/"
        f"{quote(GITHUB_REPO, safe='')}/contents"
    )

    if encoded_path:
        url = f"{url}/{encoded_path}"

    if ref:
        url = f"{url}?{urlencode({'ref': ref})}"

    return url


def _headers(*, accept: str = "application/vnd.github+json") -> dict[str, str]:
    validate_storage_configuration()

    return {
        "Accept": accept,
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "User-Agent": "Scholar-RAG-GitHub-Storage",
    }


def _decode_error_body(error: HTTPError) -> str:
    try:
        body = error.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
        return str(parsed.get("message") or parsed)
    except Exception:
        return str(error.reason or error)


def _json_request(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    expected_statuses: tuple[int, ...] = (200,),
) -> Any:
    """Send a JSON request to GitHub and decode its response."""
    data: bytes | None = None
    headers = _headers()

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        url=url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=60) as response:
            status = response.getcode()
            body = response.read()

            if status not in expected_statuses:
                raise GitHubStorageError(
                    f"GitHub returned unexpected HTTP status {status}."
                )

            if not body:
                return None

            return json.loads(body.decode("utf-8"))

    except HTTPError as exc:
        message = _decode_error_body(exc)
        raise GitHubStorageError(
            f"GitHub API error {exc.code}: {message}"
        ) from exc

    except URLError as exc:
        raise GitHubStorageError(
            f"Unable to connect to GitHub: {exc.reason}"
        ) from exc


def _raw_request(url: str) -> bytes:
    """Download raw repository file bytes through the Contents API."""
    request = Request(
        url=url,
        headers=_headers(accept="application/vnd.github.raw+json"),
        method="GET",
    )

    try:
        with urlopen(request, timeout=120) as response:
            return response.read()

    except HTTPError as exc:
        message = _decode_error_body(exc)
        raise GitHubStorageError(
            f"GitHub download error {exc.code}: {message}"
        ) from exc

    except URLError as exc:
        raise GitHubStorageError(
            f"Unable to download from GitHub: {exc.reason}"
        ) from exc


def _read_upload_bytes(file_source: Any) -> bytes:
    """
    Read data from bytes, a local path, Flask FileStorage, or a binary stream.
    """
    if isinstance(file_source, bytes):
        return file_source

    if isinstance(file_source, bytearray):
        return bytes(file_source)

    if isinstance(file_source, (str, Path)):
        path = Path(file_source)

        if not path.is_file():
            raise GitHubStorageError(f"Upload file does not exist: {path}")

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

    raise GitHubStorageError(
        "Unsupported upload source. Pass a Flask FileStorage object, "
        "a file path, bytes, or a binary file object."
    )


def get_document_details(filename: str) -> dict[str, Any] | None:
    """
    Return GitHub metadata for one document, or None when it does not exist.
    """
    repository_path = _repository_path(filename)
    url = _contents_url(repository_path, ref=GITHUB_BRANCH)

    try:
        result = _json_request("GET", url, expected_statuses=(200,))
    except GitHubStorageError as exc:
        if "GitHub API error 404:" in str(exc):
            return None
        raise

    if not isinstance(result, dict) or result.get("type") != "file":
        return None

    return result


def upload_document(
    file_source: Any,
    filename: str | None = None,
    *,
    overwrite: bool = True,
    commit_message: str | None = None,
) -> dict[str, Any]:
    """
    Create or replace a document in the configured GitHub repository.

    Returns:
        {
            "name": "...",
            "path": "documents/...",
            "sha": "...",
            "html_url": "...",
            "commit_sha": "..."
        }
    """
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
        raise GitHubStorageError("Filename could not be detected.")

    safe_name = _safe_filename(detected_name)
    repository_path = _repository_path(safe_name)
    content = _read_upload_bytes(file_source)

    if not content:
        raise GitHubStorageError("The uploaded document is empty.")

    if len(content) > GITHUB_MAX_UPLOAD_BYTES:
        raise GitHubStorageError(
            f"File is too large ({len(content):,} bytes). "
            f"Configured maximum is {GITHUB_MAX_UPLOAD_BYTES:,} bytes."
        )

    existing = get_document_details(safe_name)

    if existing and not overwrite:
        raise GitHubStorageError(
            f"'{safe_name}' already exists in GitHub."
        )

    payload: dict[str, Any] = {
        "message": (
            commit_message
            or (
                f"Update RAG document: {safe_name}"
                if existing
                else f"Add RAG document: {safe_name}"
            )
        ),
        "content": base64.b64encode(content).decode("ascii"),
        "branch": GITHUB_BRANCH,
    }

    # GitHub requires the current blob SHA when replacing an existing file.
    if existing:
        payload["sha"] = existing["sha"]

    response = _json_request(
        "PUT",
        _contents_url(repository_path),
        payload,
        expected_statuses=(200, 201),
    )

    content_data = (response or {}).get("content") or {}
    commit_data = (response or {}).get("commit") or {}

    return {
        "name": content_data.get("name", safe_name),
        "path": content_data.get("path", repository_path),
        "sha": content_data.get("sha"),
        "html_url": content_data.get("html_url"),
        "commit_sha": commit_data.get("sha"),
        "content_type": (
            mimetypes.guess_type(safe_name)[0]
            or "application/octet-stream"
        ),
    }


def list_document_details() -> list[dict[str, Any]]:
    """List supported documents and their GitHub metadata."""
    path = GITHUB_DOCUMENTS_PATH
    url = _contents_url(path, ref=GITHUB_BRANCH)

    try:
        response = _json_request("GET", url, expected_statuses=(200,))
    except GitHubStorageError as exc:
        # A missing documents directory is equivalent to an empty store.
        if "GitHub API error 404:" in str(exc):
            return []
        raise

    if not isinstance(response, list):
        return []

    documents: list[dict[str, Any]] = []

    for item in response:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name") or "").strip()

        if (
            item.get("type") == "file"
            and Path(name).suffix.lower() in ALLOWED_DOCUMENT_EXTENSIONS
        ):
            documents.append(
                {
                    "name": name,
                    "path": item.get("path"),
                    "sha": item.get("sha"),
                    "size": item.get("size", 0),
                    "html_url": item.get("html_url"),
                    "download_url": item.get("download_url"),
                }
            )

    return sorted(documents, key=lambda item: item["name"].lower())


def list_documents() -> list[str]:
    """Return only stored document filenames."""
    return [item["name"] for item in list_document_details()]


def download_document(
    filename: str,
    destination_dir: str | Path = DOCUMENTS_DIR,
) -> Path:
    """Download one GitHub document into the local documents directory."""
    safe_name = _safe_filename(filename)
    repository_path = _repository_path(safe_name)

    data = _raw_request(
        _contents_url(repository_path, ref=GITHUB_BRANCH)
    )

    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)

    local_path = destination / safe_name
    local_path.write_bytes(data)

    return local_path


def download_all_documents(
    destination_dir: str | Path = DOCUMENTS_DIR,
    *,
    clear_local: bool = False,
) -> list[Path]:
    """Download every supported GitHub document to a local directory."""
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)

    if clear_local:
        for local_file in destination.iterdir():
            if (
                local_file.is_file()
                and local_file.suffix.lower()
                in ALLOWED_DOCUMENT_EXTENSIONS
            ):
                local_file.unlink()

    downloaded: list[Path] = []

    for filename in list_documents():
        downloaded.append(download_document(filename, destination))

    return downloaded


def sync_documents_to_local(
    destination_dir: str | Path = DOCUMENTS_DIR,
) -> list[Path]:
    """
    Make the local documents folder match the GitHub documents folder.

    Call this before running ingest_documents() on Render.
    """
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)

    remote_names = set(list_documents())

    for local_file in destination.iterdir():
        if (
            local_file.is_file()
            and local_file.suffix.lower()
            in ALLOWED_DOCUMENT_EXTENSIONS
            and local_file.name not in remote_names
        ):
            local_file.unlink()

    downloaded: list[Path] = []

    for filename in sorted(remote_names, key=str.lower):
        downloaded.append(download_document(filename, destination))

    return downloaded


def delete_document(
    filename: str,
    *,
    commit_message: str | None = None,
) -> dict[str, Any]:
    """Delete one document from GitHub and commit the deletion."""
    safe_name = _safe_filename(filename)
    existing = get_document_details(safe_name)

    if not existing:
        raise GitHubStorageError(
            f"'{safe_name}' was not found in GitHub."
        )

    repository_path = _repository_path(safe_name)

    payload = {
        "message": (
            commit_message
            or f"Delete RAG document: {safe_name}"
        ),
        "sha": existing["sha"],
        "branch": GITHUB_BRANCH,
    }

    response = _json_request(
        "DELETE",
        _contents_url(repository_path),
        payload,
        expected_statuses=(200,),
    )

    commit_data = (response or {}).get("commit") or {}

    return {
        "name": safe_name,
        "path": repository_path,
        "deleted": True,
        "commit_sha": commit_data.get("sha"),
    }


def document_exists(filename: str) -> bool:
    """Return True when a document exists in GitHub."""
    return get_document_details(filename) is not None


def get_storage_status() -> dict[str, Any]:
    """Return connection and document-count information."""
    status: dict[str, Any] = {
        "configured": is_storage_configured(),
        "owner": GITHUB_OWNER,
        "repository": GITHUB_REPO,
        "branch": GITHUB_BRANCH,
        "documents_path": GITHUB_DOCUMENTS_PATH,
        "document_count": 0,
        "error": None,
    }

    if not status["configured"]:
        status["error"] = (
            "GitHub environment variables are incomplete."
        )
        return status

    try:
        status["document_count"] = len(list_documents())
    except GitHubStorageError as exc:
        status["error"] = str(exc)

    return status


if __name__ == "__main__":
    print("GitHub Storage status:")

    for key, value in get_storage_status().items():
        print(f"  {key}: {value}")

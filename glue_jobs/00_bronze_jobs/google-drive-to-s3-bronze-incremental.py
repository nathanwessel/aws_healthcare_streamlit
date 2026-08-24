"""AWS Glue PySpark job: incrementally mirror a Google Drive folder into S3.

Design
------
* First run (bootstrap):
    1. Get a Google Drive Changes API start page token.
    2. Recursively crawl the configured Drive folder (metadata only).
    3. Download every normal file and create zero-byte S3 folder markers.
    4. Save manifest.json, then page_token.json.

* Later runs:
    1. Read the saved page token and manifest from S3.
    2. Read Google Drive Changes since the saved token.
    3. If changes exist, recursively crawl the target Drive folder (metadata only).
    4. Diff the old/current manifests by Google file ID.
    5. Compare Drive content checksums so metadata-only renames/moves use S3 copies,
       download only new/content-changed files, create/delete folder markers,
       and delete objects that left the Drive tree.
    6. Only after every required action succeeds, save manifest.json and advance
       page_token.json to Google's newStartPageToken.

This job intentionally uses Spark/Glue as the managed runtime/orchestration
container. The Google Drive -> S3 mirror work runs on the Glue driver as Python.

Required Glue job parameters:
    --GOOGLE_DRIVE_FOLDER_ID
    --S3_BUCKET
    --S3_PREFIX
    --GOOGLE_SECRET_NAME
    --STATE_S3_PREFIX

Example values:
    --GOOGLE_DRIVE_FOLDER_ID  1jSf1kZ6ML9LgWfQ0WLWLVSe26NE-c6iT
    --S3_BUCKET               nw-healthcare-project
    --S3_PREFIX               bronze/
    --GOOGLE_SECRET_NAME      google-drive-oauth-token-glue-job
    --STATE_S3_PREFIX         _control/google-drive/

Expected Secrets Manager shape:
    Secret name: google-drive-oauth-token-glue-job
    Secret JSON:
    {
      "google-drive-oauth-token": "<the COMPLETE contents of local token.json>"
    }

The secret value may also be a nested JSON object rather than a JSON string.
"""

import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import PurePosixPath

import boto3
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext


AWS_REGION = "us-east-1"
GOOGLE_SECRET_KEY = "google-drive-oauth-token"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
SHORTCUT_MIME_TYPE = "application/vnd.google-apps.shortcut"
GOOGLE_NATIVE_MIME_PREFIX = "application/vnd.google-apps."

DRIVE_DOWNLOAD_CHUNK_SIZE = 10 * 1024 * 1024
MANIFEST_SCHEMA_VERSION = 2

S3_TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=64 * 1024 * 1024,
    multipart_chunksize=64 * 1024 * 1024,
    max_concurrency=4,
    use_threads=True,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_s3_prefix(prefix: str) -> str:
    """Return an S3 prefix with no leading slash and exactly one trailing slash."""
    prefix = prefix.strip().strip("/")
    return f"{prefix}/" if prefix else ""


def sanitize_drive_name(name: str) -> str:
    """Prevent Drive names from accidentally creating extra S3 path levels."""
    return name.replace("/", "_").replace("\\", "_")


def make_s3_key(s3_prefix: str, relative_parts: tuple[str, ...]) -> str:
    relative_key = str(PurePosixPath(*relative_parts))
    return f"{s3_prefix}{relative_key}"


def make_folder_s3_key(s3_prefix: str, relative_parts: tuple[str, ...]) -> str:
    return make_s3_key(s3_prefix, relative_parts).rstrip("/") + "/"


def relative_path_string(relative_parts: tuple[str, ...], is_folder: bool) -> str:
    value = str(PurePosixPath(*relative_parts))
    return value.rstrip("/") + "/" if is_folder else value


def state_object_keys(state_prefix: str, folder_id: str) -> dict:
    root = f"{normalize_s3_prefix(state_prefix)}{folder_id}/"
    return {
        "root": root,
        "page_token": f"{root}page_token.json",
        "manifest": f"{root}manifest.json",
        "move_staging": f"{root}_move_staging/",
    }


# ---------------------------------------------------------------------------
# Google authentication / Drive helpers
# ---------------------------------------------------------------------------


def load_google_credentials(secret_name: str) -> Credentials:
    secrets_client = boto3.client("secretsmanager", region_name=AWS_REGION)
    response = secrets_client.get_secret_value(SecretId=secret_name)

    if "SecretString" not in response:
        raise ValueError(
            f"Secret {secret_name!r} is binary. Store the Google token as SecretString JSON."
        )

    outer_secret = json.loads(response["SecretString"])
    if GOOGLE_SECRET_KEY not in outer_secret:
        raise KeyError(
            f"Secret {secret_name!r} does not contain key {GOOGLE_SECRET_KEY!r}."
        )

    token_info = outer_secret[GOOGLE_SECRET_KEY]
    if isinstance(token_info, str):
        token_info = json.loads(token_info)

    if not isinstance(token_info, dict):
        raise ValueError(
            f"Secret key {GOOGLE_SECRET_KEY!r} must contain the complete token.json JSON."
        )

    required_fields = {"refresh_token", "token_uri", "client_id", "client_secret"}
    missing = sorted(field for field in required_fields if not token_info.get(field))
    if missing:
        raise ValueError("Google OAuth secret is incomplete. Missing: " + ", ".join(missing))

    credentials = Credentials.from_authorized_user_info(token_info, SCOPES)
    if not credentials.valid:
        print("[AUTH] Refreshing Google OAuth access token...")
        credentials.refresh(Request())

    return credentials


def verify_root_folder(drive_service, folder_id: str) -> dict:
    """Fail safely if the configured root folder is missing, trashed, or not a folder."""
    root = (
        drive_service.files()
        .get(fileId=folder_id, fields="id,name,mimeType,trashed")
        .execute()
    )

    if root.get("mimeType") != FOLDER_MIME_TYPE:
        raise ValueError(f"Configured Google Drive ID {folder_id!r} is not a folder.")
    if root.get("trashed"):
        raise ValueError(
            "Configured Google Drive root folder is trashed. Refusing to treat the whole S3 mirror as deleted."
        )
    return root


def list_folder_items(drive_service, folder_id: str):
    """Yield immediate non-trashed children of a Drive folder."""
    page_token = None
    while True:
        response = (
            drive_service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                pageSize=1000,
                pageToken=page_token,
                fields=(
                    "nextPageToken, "
                    "files(id,name,mimeType,size,parents,modifiedTime,md5Checksum,capabilities(canDownload))"
                ),
            )
            .execute()
        )

        yield from response.get("files", [])
        page_token = response.get("nextPageToken")
        if not page_token:
            break


def get_start_page_token(drive_service) -> str:
    response = drive_service.changes().getStartPageToken().execute()
    token = response.get("startPageToken")
    if not token:
        raise RuntimeError("Google Drive did not return startPageToken.")
    return token


def get_changes_since(drive_service, saved_page_token: str) -> tuple[set[str], str, int]:
    """Return (changed_file_ids, new_start_page_token, total_change_records)."""
    changed_ids: set[str] = set()
    total_changes = 0
    page_token = saved_page_token
    new_start_page_token = None

    while True:
        response = (
            drive_service.changes()
            .list(
                pageToken=page_token,
                pageSize=1000,
                spaces="drive",
                includeRemoved=True,
                fields=(
                    "nextPageToken,newStartPageToken,"
                    "changes(fileId,removed,time)"
                ),
            )
            .execute()
        )

        changes = response.get("changes", [])
        total_changes += len(changes)
        changed_ids.update(change["fileId"] for change in changes if change.get("fileId"))

        next_page = response.get("nextPageToken")
        if next_page:
            page_token = next_page
            continue

        new_start_page_token = response.get("newStartPageToken")
        break

    if not new_start_page_token:
        raise RuntimeError(
            "Google Drive changes.list reached the final page without newStartPageToken."
        )

    return changed_ids, new_start_page_token, total_changes


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------


def build_drive_manifest(
    drive_service,
    root_folder_id: str,
    s3_prefix: str,
    log_skips: bool,
) -> tuple[dict, dict]:
    """Build current Drive-tree metadata without downloading file contents.

    The root Drive folder itself is intentionally excluded. Subfolders are
    included as zero-byte S3 folder-marker entries so empty folders are kept.
    Google-native files and shortcuts are excluded from the managed manifest.
    """
    verify_root_folder(drive_service, root_folder_id)

    manifest: dict[str, dict] = {}
    seen_s3_keys: dict[str, str] = {}
    crawl_stats = {"folders": 0, "files": 0, "skipped": 0}

    def add_entry(file_id: str, entry: dict):
        s3_key = entry["s3_key"]
        other_id = seen_s3_keys.get(s3_key)
        if other_id and other_id != file_id:
            raise ValueError(
                "Two Google Drive items map to the same S3 key. "
                f"S3 cannot safely mirror duplicate names at {s3_key!r}. "
                f"Drive IDs: {other_id!r} and {file_id!r}."
            )
        seen_s3_keys[s3_key] = file_id
        manifest[file_id] = entry

    def walk(folder_id: str, relative_parts: tuple[str, ...]):
        for item in list_folder_items(drive_service, folder_id):
            item_id = item["id"]
            item_name = sanitize_drive_name(item["name"])
            mime_type = item.get("mimeType", "")
            child_parts = (*relative_parts, item_name)
            parent_ids = item.get("parents", [])
            parent_id = parent_ids[0] if parent_ids else None

            if mime_type == FOLDER_MIME_TYPE:
                s3_key = make_folder_s3_key(s3_prefix, child_parts)
                add_entry(
                    item_id,
                    {
                        "type": "folder",
                        "name": item_name,
                        "relative_path": relative_path_string(child_parts, True),
                        "s3_key": s3_key,
                        "parent_id": parent_id,
                        "mime_type": mime_type,
                        "modified_time": item.get("modifiedTime"),
                    },
                )
                crawl_stats["folders"] += 1
                walk(item_id, child_parts)
                continue

            if mime_type == SHORTCUT_MIME_TYPE:
                crawl_stats["skipped"] += 1
                if log_skips:
                    print(f"[SKIP] Shortcut: {'/'.join(child_parts)}")
                continue

            if mime_type.startswith(GOOGLE_NATIVE_MIME_PREFIX):
                crawl_stats["skipped"] += 1
                if log_skips:
                    print(f"[SKIP] Google-native file: {'/'.join(child_parts)} ({mime_type})")
                continue

            if not item.get("capabilities", {}).get("canDownload", True):
                crawl_stats["skipped"] += 1
                if log_skips:
                    print(f"[SKIP] Download not permitted: {'/'.join(child_parts)}")
                continue

            s3_key = make_s3_key(s3_prefix, child_parts)
            add_entry(
                item_id,
                {
                    "type": "file",
                    "name": item_name,
                    "relative_path": relative_path_string(child_parts, False),
                    "s3_key": s3_key,
                    "parent_id": parent_id,
                    "mime_type": mime_type,
                    "size": item.get("size"),
                    "modified_time": item.get("modifiedTime"),
                    # Google Drive exposes md5Checksum for blob/binary file content.
                    # We store it so metadata-only changes such as rename/move do
                    # not force an unnecessary re-download from Google.
                    "md5_checksum": item.get("md5Checksum"),
                },
            )
            crawl_stats["files"] += 1

    walk(root_folder_id, ())
    return manifest, crawl_stats


# ---------------------------------------------------------------------------
# S3 state / object helpers
# ---------------------------------------------------------------------------


def load_json_from_s3(s3_client, bucket: str, key: str):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code")
        if code in {"NoSuchKey", "404"}:
            return None
        raise


def save_json_to_s3(s3_client, bucket: str, key: str, payload: dict):
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def create_folder_marker(s3_client, bucket: str, key: str):
    s3_client.put_object(Bucket=bucket, Key=key, Body=b"")


def delete_s3_object(s3_client, bucket: str, key: str):
    s3_client.delete_object(Bucket=bucket, Key=key)


def copy_s3_object(s3_client, bucket: str, source_key: str, destination_key: str):
    s3_client.copy_object(
        Bucket=bucket,
        Key=destination_key,
        CopySource={"Bucket": bucket, "Key": source_key},
    )


# ---------------------------------------------------------------------------
# Drive file download
# ---------------------------------------------------------------------------


def download_drive_file_to_s3(
    drive_service,
    s3_client,
    file_id: str,
    display_name: str,
    s3_bucket: str,
    s3_key: str,
) -> bool:
    """Download one normal Drive file to /tmp in chunks, then upload to S3."""
    temp_path = None
    try:
        request = drive_service.files().get_media(fileId=file_id)

        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="google_drive_",
            suffix=".part",
            dir="/tmp",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            downloader = MediaIoBaseDownload(
                temp_file,
                request,
                chunksize=DRIVE_DOWNLOAD_CHUNK_SIZE,
            )

            print(f"[DOWNLOAD] gdrive://{display_name} -> s3://{s3_bucket}/{s3_key}")
            done = False
            last_percent = -1
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    percent = int(status.progress() * 100)
                    if percent != last_percent:
                        print(f"           Google download: {percent:3d}%")
                        last_percent = percent

        s3_client.upload_file(
            Filename=temp_path,
            Bucket=s3_bucket,
            Key=s3_key,
            Config=S3_TRANSFER_CONFIG,
        )
        print(f"[DONE]     s3://{s3_bucket}/{s3_key}")
        return True

    except Exception as error:
        print(f"[ERROR] Download failed for {display_name}: {error}")
        return False
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as cleanup_error:
                print(f"[WARN] Could not delete temp file {temp_path}: {cleanup_error}")


def file_content_requires_download(old_entry: dict, new_entry: dict, drive_id_changed: bool) -> bool:
    """Return True only when a changed Drive file may have different bytes.

    For normal blob files, matching non-empty Drive MD5 checksums mean the file
    content is unchanged even if metadata such as the name or parent changed.
    If either checksum is unavailable, fall back to downloading for correctness.
    """
    if not drive_id_changed:
        return False

    old_md5 = old_entry.get("md5_checksum")
    new_md5 = new_entry.get("md5_checksum")
    if old_md5 and new_md5 and old_md5 == new_md5:
        return False

    return True


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_sync(
    drive_service,
    s3_client,
    folder_id: str,
    s3_bucket: str,
    s3_prefix: str,
    state_keys: dict,
) -> dict:
    """Full-sync bootstrap. State is created only after every action succeeds."""
    print("[BOOTSTRAP] No saved incremental state found.")
    print("[BOOTSTRAP] Getting Changes API start page token BEFORE the full sync...")
    start_page_token = get_start_page_token(drive_service)

    print("[BOOTSTRAP] Building current Google Drive manifest...")
    current_manifest, crawl_stats = build_drive_manifest(
        drive_service=drive_service,
        root_folder_id=folder_id,
        s3_prefix=s3_prefix,
        log_skips=True,
    )

    stats = {
        "mode": "bootstrap",
        "downloaded": 0,
        "moved": 0,
        "deleted": 0,
        "folder_markers_created": 0,
        "failed": 0,
        "drive_change_records": 0,
        "crawl": crawl_stats,
    }

    # Folder markers first, shallowest first, then files.
    folder_entries = sorted(
        (entry for entry in current_manifest.values() if entry["type"] == "folder"),
        key=lambda entry: entry["relative_path"].count("/"),
    )
    file_entries = [
        (file_id, entry)
        for file_id, entry in current_manifest.items()
        if entry["type"] == "file"
    ]

    for entry in folder_entries:
        try:
            create_folder_marker(s3_client, s3_bucket, entry["s3_key"])
            print(f"[FOLDER]   s3://{s3_bucket}/{entry['s3_key']}")
            stats["folder_markers_created"] += 1
        except Exception as error:
            print(f"[ERROR] Could not create folder marker {entry['s3_key']}: {error}")
            stats["failed"] += 1

    for file_id, entry in file_entries:
        success = download_drive_file_to_s3(
            drive_service=drive_service,
            s3_client=s3_client,
            file_id=file_id,
            display_name=entry["relative_path"],
            s3_bucket=s3_bucket,
            s3_key=entry["s3_key"],
        )
        if success:
            stats["downloaded"] += 1
        else:
            stats["failed"] += 1

    if stats["failed"]:
        raise RuntimeError(
            f"Bootstrap attempted all items but had {stats['failed']} failure(s). "
            "Incremental state was NOT created; the next run will bootstrap again."
        )

    manifest_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "folder_id": folder_id,
        "generated_at": utc_now_iso(),
        "items": current_manifest,
    }
    page_token_payload = {
        "folder_id": folder_id,
        "page_token": start_page_token,
        "last_successful_run": utc_now_iso(),
        "bootstrap_completed": True,
    }

    # Save manifest first and page token last. If the token write fails, the old
    # token (or no token during bootstrap) is not advanced past unsaved state.
    save_json_to_s3(s3_client, s3_bucket, state_keys["manifest"], manifest_payload)
    save_json_to_s3(s3_client, s3_bucket, state_keys["page_token"], page_token_payload)

    print(f"[STATE] Saved s3://{s3_bucket}/{state_keys['manifest']}")
    print(f"[STATE] Saved s3://{s3_bucket}/{state_keys['page_token']}")
    return stats


# ---------------------------------------------------------------------------
# Incremental manifest reconciliation
# ---------------------------------------------------------------------------


def incremental_sync(
    drive_service,
    s3_client,
    folder_id: str,
    s3_bucket: str,
    s3_prefix: str,
    state_keys: dict,
    page_token_payload: dict,
    manifest_payload: dict,
) -> dict:
    old_manifest = manifest_payload.get("items", {})
    saved_page_token = page_token_payload.get("page_token")

    if not saved_page_token:
        raise ValueError("page_token.json exists but does not contain page_token.")
    if manifest_payload.get("folder_id") != folder_id or page_token_payload.get("folder_id") != folder_id:
        raise ValueError("Saved incremental state belongs to a different Google Drive folder ID.")

    print(f"[CHANGES] Reading Google Drive changes since token {saved_page_token[:12]}...")
    try:
        changed_ids, new_page_token, total_change_records = get_changes_since(
            drive_service, saved_page_token
        )
    except HttpError as error:
        if getattr(error, "resp", None) is not None and error.resp.status == 410:
            raise RuntimeError(
                "Google rejected the saved Changes API page token (HTTP 410). "
                "Do not advance state automatically. After verifying S3, delete this folder's "
                "page_token.json and manifest.json to force a clean bootstrap."
            ) from error
        raise

    stats = {
        "mode": "incremental",
        "downloaded": 0,
        "moved": 0,
        "deleted": 0,
        "folder_markers_created": 0,
        "failed": 0,
        "drive_change_records": total_change_records,
        "crawl": None,
    }

    if total_change_records == 0:
        print("[CHANGES] No Google Drive changes found.")

        # One-time, metadata-only migration for manifests created by the earlier
        # incremental script, which did not persist Drive content checksums.
        # This does NOT re-download files and does NOT require a new bootstrap.
        if manifest_payload.get("schema_version", 1) < MANIFEST_SCHEMA_VERSION:
            print("[STATE] Upgrading manifest schema to store Drive content checksums...")
            current_manifest, crawl_stats = build_drive_manifest(
                drive_service=drive_service,
                root_folder_id=folder_id,
                s3_prefix=s3_prefix,
                log_skips=False,
            )
            stats["crawl"] = crawl_stats
            upgraded_manifest_payload = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "folder_id": folder_id,
                "generated_at": utc_now_iso(),
                "items": current_manifest,
            }
            save_json_to_s3(
                s3_client, s3_bucket, state_keys["manifest"], upgraded_manifest_payload
            )
            print("[STATE] Manifest schema upgraded without downloading file contents.")

        updated_page_token_payload = {
            "folder_id": folder_id,
            "page_token": new_page_token,
            "last_successful_run": utc_now_iso(),
            "bootstrap_completed": True,
        }
        save_json_to_s3(
            s3_client, s3_bucket, state_keys["page_token"], updated_page_token_payload
        )
        print("[STATE] Advanced to Google's latest page token.")
        return stats

    print(f"[CHANGES] {total_change_records} change record(s), {len(changed_ids)} unique Drive ID(s).")
    print("[MANIFEST] Crawling configured Drive folder metadata...")
    current_manifest, crawl_stats = build_drive_manifest(
        drive_service=drive_service,
        root_folder_id=folder_id,
        s3_prefix=s3_prefix,
        log_skips=False,
    )
    stats["crawl"] = crawl_stats

    old_ids = set(old_manifest)
    current_ids = set(current_manifest)
    removed_ids = old_ids - current_ids
    added_ids = current_ids - old_ids
    common_ids = old_ids & current_ids

    moved_ids = {
        file_id
        for file_id in common_ids
        if old_manifest[file_id].get("s3_key") != current_manifest[file_id].get("s3_key")
    }

    same_path_changed_file_ids = {
        file_id
        for file_id in common_ids
        if file_id in changed_ids
        and file_id not in moved_ids
        and current_manifest[file_id].get("type") == "file"
    }

    same_path_content_changed_file_ids = {
        file_id
        for file_id in same_path_changed_file_ids
        if file_content_requires_download(
            old_manifest[file_id], current_manifest[file_id], drive_id_changed=True
        )
    }
    same_path_metadata_only_file_ids = (
        same_path_changed_file_ids - same_path_content_changed_file_ids
    )

    print(
        "[PLAN] "
        f"new={len(added_ids)}, "
        f"moved/renamed={len(moved_ids)}, "
        f"removed={len(removed_ids)}, "
        f"content-changed-in-place={len(same_path_content_changed_file_ids)}, "
        f"metadata-only-in-place={len(same_path_metadata_only_file_ids)}"
    )

    # Drive changes elsewhere in the user's account can produce a nonzero feed
    # even when nothing in this configured tree needs action. We intentionally
    # do not log each irrelevant Drive change.
    if not (added_ids or moved_ids or removed_ids or same_path_content_changed_file_ids):
        manifest_payload_out = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "folder_id": folder_id,
            "generated_at": utc_now_iso(),
            "items": current_manifest,
        }
        page_token_payload_out = {
            "folder_id": folder_id,
            "page_token": new_page_token,
            "last_successful_run": utc_now_iso(),
            "bootstrap_completed": True,
        }
        save_json_to_s3(s3_client, s3_bucket, state_keys["manifest"], manifest_payload_out)
        save_json_to_s3(s3_client, s3_bucket, state_keys["page_token"], page_token_payload_out)
        print("[CHANGES] No applicable changes under configured Drive folder.")
        return stats

    # Current S3 keys are protected from cleanup. This matters for swaps/reuse,
    # where one item's new key can equal another item's old key.
    current_s3_keys = {entry["s3_key"] for entry in current_manifest.values()}

    # For path-only file moves caused by an ancestor folder rename/move, stage
    # the old S3 object before writing any final destinations. This prevents
    # source-overwrite problems if paths are swapped/reused in one batch.
    run_id = uuid.uuid4().hex
    staged_move_keys: dict[str, str] = {}
    move_staging_failures: set[str] = set()

    for file_id in sorted(moved_ids):
        old_entry = old_manifest[file_id]
        new_entry = current_manifest[file_id]

        if new_entry.get("type") != "file":
            continue
        if file_content_requires_download(
            old_entry, new_entry, drive_id_changed=(file_id in changed_ids)
        ):
            # The bytes may have changed, so download the current content rather
            # than copying a potentially stale object from the old S3 key.
            continue

        staging_key = f"{state_keys['move_staging']}{run_id}/{file_id}"
        try:
            copy_s3_object(
                s3_client=s3_client,
                bucket=s3_bucket,
                source_key=old_entry["s3_key"],
                destination_key=staging_key,
            )
            staged_move_keys[file_id] = staging_key
        except Exception as error:
            print(
                f"[ERROR] Could not stage move source s3://{s3_bucket}/{old_entry['s3_key']}: {error}"
            )
            move_staging_failures.add(file_id)
            stats["failed"] += 1

    successful_moved_ids: set[str] = set()

    # 1) New folders and moved/renamed folders: create current marker.
    folder_upsert_ids = {
        file_id
        for file_id in (added_ids | moved_ids)
        if current_manifest[file_id].get("type") == "folder"
    }
    for file_id in sorted(folder_upsert_ids):
        entry = current_manifest[file_id]
        try:
            create_folder_marker(s3_client, s3_bucket, entry["s3_key"])
            if file_id in moved_ids:
                old_key = old_manifest[file_id]["s3_key"]
                print(f"[MOVE]     {old_key} -> {entry['s3_key']}")
                successful_moved_ids.add(file_id)
                stats["moved"] += 1
            else:
                print(f"[FOLDER]   s3://{s3_bucket}/{entry['s3_key']}")
                stats["folder_markers_created"] += 1
        except Exception as error:
            print(f"[ERROR] Folder marker failed for {entry['s3_key']}: {error}")
            stats["failed"] += 1

    # 2) New files: always download.
    for file_id in sorted(added_ids):
        entry = current_manifest[file_id]
        if entry.get("type") != "file":
            continue
        if download_drive_file_to_s3(
            drive_service,
            s3_client,
            file_id,
            entry["relative_path"],
            s3_bucket,
            entry["s3_key"],
        ):
            stats["downloaded"] += 1
        else:
            stats["failed"] += 1

    # 3) Files moved/renamed. If the file ID itself changed, download fresh;
    # otherwise promote its staged S3 copy.
    for file_id in sorted(moved_ids):
        new_entry = current_manifest[file_id]
        old_entry = old_manifest[file_id]
        if new_entry.get("type") != "file":
            continue

        if file_id in move_staging_failures:
            continue

        success = False
        if file_content_requires_download(
            old_entry, new_entry, drive_id_changed=(file_id in changed_ids)
        ):
            success = download_drive_file_to_s3(
                drive_service,
                s3_client,
                file_id,
                new_entry["relative_path"],
                s3_bucket,
                new_entry["s3_key"],
            )
            if success:
                stats["downloaded"] += 1
        else:
            staging_key = staged_move_keys[file_id]
            try:
                copy_s3_object(
                    s3_client=s3_client,
                    bucket=s3_bucket,
                    source_key=staging_key,
                    destination_key=new_entry["s3_key"],
                )
                success = True
            except Exception as error:
                print(
                    f"[ERROR] Could not finish move to s3://{s3_bucket}/{new_entry['s3_key']}: {error}"
                )

        if success:
            print(f"[MOVE]     {old_entry['s3_key']} -> {new_entry['s3_key']}")
            successful_moved_ids.add(file_id)
            stats["moved"] += 1
        else:
            stats["failed"] += 1

    # 4) Same-path files whose content checksum changed (or was unavailable): overwrite.
    # Metadata-only changes with matching checksums need no S3 data movement.
    for file_id in sorted(same_path_content_changed_file_ids):
        entry = current_manifest[file_id]
        if download_drive_file_to_s3(
            drive_service,
            s3_client,
            file_id,
            entry["relative_path"],
            s3_bucket,
            entry["s3_key"],
        ):
            stats["downloaded"] += 1
        else:
            stats["failed"] += 1

    # 5) Delete objects that no longer belong in the current Drive tree.
    # For moved items, only delete the old key if the replacement succeeded.
    cleanup_candidates: list[tuple[str, str, str]] = []
    for file_id in removed_ids:
        cleanup_candidates.append((file_id, old_manifest[file_id]["s3_key"], "removed"))
    for file_id in successful_moved_ids:
        cleanup_candidates.append((file_id, old_manifest[file_id]["s3_key"], "moved"))

    for file_id, old_key, reason in cleanup_candidates:
        if old_key in current_s3_keys:
            # Another current Drive item now legitimately owns this exact key.
            continue
        try:
            delete_s3_object(s3_client, s3_bucket, old_key)
            if reason == "removed":
                print(f"[DELETE]   s3://{s3_bucket}/{old_key}")
                stats["deleted"] += 1
        except Exception as error:
            print(f"[ERROR] Could not delete s3://{s3_bucket}/{old_key}: {error}")
            stats["failed"] += 1

    # Best-effort cleanup of temporary move staging objects. These are control
    # artifacts only, so a cleanup warning does not invalidate the bronze sync.
    for staging_key in staged_move_keys.values():
        try:
            delete_s3_object(s3_client, s3_bucket, staging_key)
        except Exception as error:
            print(f"[WARN] Could not remove move staging object {staging_key}: {error}")

    if stats["failed"]:
        raise RuntimeError(
            f"Incremental reconciliation attempted all applicable changes but had "
            f"{stats['failed']} failure(s). manifest.json and page_token.json were NOT advanced. "
            "The next run will replay the same Google Drive change window."
        )

    manifest_payload_out = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "folder_id": folder_id,
        "generated_at": utc_now_iso(),
        "items": current_manifest,
    }
    page_token_payload_out = {
        "folder_id": folder_id,
        "page_token": new_page_token,
        "last_successful_run": utc_now_iso(),
        "bootstrap_completed": True,
    }

    # Save manifest first, page token second. The token is the commit point.
    save_json_to_s3(s3_client, s3_bucket, state_keys["manifest"], manifest_payload_out)
    save_json_to_s3(s3_client, s3_bucket, state_keys["page_token"], page_token_payload_out)
    print("[STATE] Manifest saved; Google Drive page token advanced.")
    return stats


# ---------------------------------------------------------------------------
# Orchestration entry point
# ---------------------------------------------------------------------------


def mirror_google_drive_folder_incrementally(
    folder_id: str,
    s3_bucket: str,
    s3_prefix: str,
    google_secret_name: str,
    state_s3_prefix: str,
) -> dict:
    s3_prefix = normalize_s3_prefix(s3_prefix)
    state_s3_prefix = normalize_s3_prefix(state_s3_prefix)
    state_keys = state_object_keys(state_s3_prefix, folder_id)

    credentials = load_google_credentials(google_secret_name)
    drive_service = build(
        "drive",
        "v3",
        credentials=credentials,
        cache_discovery=False,
    )
    s3_client = boto3.client("s3", region_name=AWS_REGION)

    print(f"Google Drive folder ID: {folder_id}")
    print(f"S3 mirror:              s3://{s3_bucket}/{s3_prefix}")
    print(f"State:                  s3://{s3_bucket}/{state_keys['root']}")
    print()

    page_token_payload = load_json_from_s3(s3_client, s3_bucket, state_keys["page_token"])
    manifest_payload = load_json_from_s3(s3_client, s3_bucket, state_keys["manifest"])

    if page_token_payload is None and manifest_payload is None:
        return bootstrap_sync(
            drive_service=drive_service,
            s3_client=s3_client,
            folder_id=folder_id,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            state_keys=state_keys,
        )

    if (page_token_payload is None) != (manifest_payload is None):
        raise RuntimeError(
            "Incremental state is inconsistent: exactly one of page_token.json / manifest.json exists. "
            "Do not continue automatically. Inspect the control prefix, then restore both files or remove "
            "both to intentionally force a fresh bootstrap."
        )

    return incremental_sync(
        drive_service=drive_service,
        s3_client=s3_client,
        folder_id=folder_id,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        state_keys=state_keys,
        page_token_payload=page_token_payload,
        manifest_payload=manifest_payload,
    )


def main():
    args = getResolvedOptions(
        sys.argv,
        [
            "JOB_NAME",
            "GOOGLE_DRIVE_FOLDER_ID",
            "S3_BUCKET",
            "S3_PREFIX",
            "GOOGLE_SECRET_NAME",
            "STATE_S3_PREFIX",
        ],
    )

    spark_context = SparkContext.getOrCreate()
    glue_context = GlueContext(spark_context)
    _spark = glue_context.spark_session

    job = Job(glue_context)
    job.init(args["JOB_NAME"], args)

    stats = mirror_google_drive_folder_incrementally(
        folder_id=args["GOOGLE_DRIVE_FOLDER_ID"],
        s3_bucket=args["S3_BUCKET"],
        s3_prefix=args["S3_PREFIX"],
        google_secret_name=args["GOOGLE_SECRET_NAME"],
        state_s3_prefix=args["STATE_S3_PREFIX"],
    )

    print("\nRun complete.")
    print(f"Mode:                   {stats['mode']}")
    print(f"Drive change records:   {stats['drive_change_records']}")
    print(f"Files downloaded:       {stats['downloaded']}")
    print(f"Items moved/renamed:    {stats['moved']}")
    print(f"S3 objects deleted:     {stats['deleted']}")
    print(f"Folder markers created: {stats['folder_markers_created']}")
    print(f"Failures:               {stats['failed']}")

    if stats.get("crawl"):
        print(f"Drive folders crawled:  {stats['crawl']['folders']}")
        print(f"Drive files discovered: {stats['crawl']['files']}")
        print(f"Drive items skipped:    {stats['crawl']['skipped']}")

    job.commit()


if __name__ == "__main__":
    main()
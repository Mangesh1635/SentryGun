"""Evidence clips: local mp4 while recording, Cloudinary on finish.

Credentials come ONLY from environment variables (or a local .env
file read by core.load_dotenv): CLOUDINARY_CLOUD_NAME,
CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET. Nothing is hardcoded.
If Cloudinary is not configured, or an upload fails, the clip stays
local and the dashboard still lists it, marked local-only. The
stream never crashes because an upload failed.
"""

from __future__ import annotations

import json
import os
import time


def upload_to_cloudinary(local_path: str):
    """Returns the secure URL, or None if not configured / failed."""
    cloud = os.environ.get("CLOUDINARY_CLOUD_NAME")
    key = os.environ.get("CLOUDINARY_API_KEY")
    secret = os.environ.get("CLOUDINARY_API_SECRET")
    if not (cloud and key and secret):
        return None
    try:
        import cloudinary
        import cloudinary.uploader

        cloudinary.config(cloud_name=cloud, api_key=key,
                          api_secret=secret, secure=True)
        result = cloudinary.uploader.upload(
            local_path, resource_type="video",
            folder="sentrygun/evidence")
        return result.get("secure_url")
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[WARN] Cloudinary upload failed: {exc}")
        return None


def record_evidence(evidence_dir: str, entries: list, local_path: str):
    """Finish one clip: upload if possible, append to the JSON list
    the dashboard reads. Returns the entry dict."""
    url = upload_to_cloudinary(local_path)
    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "local_file": os.path.basename(local_path),
        "cloudinary_url": url,
        "status": "cloudinary" if url else "local-only",
    }
    entries.append(entry)
    os.makedirs(evidence_dir, exist_ok=True)
    with open(os.path.join(evidence_dir, "evidence.json"), "w",
              encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)
    return entry


def load_evidence(evidence_dir: str) -> list:
    path = os.path.join(evidence_dir, "evidence.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return []

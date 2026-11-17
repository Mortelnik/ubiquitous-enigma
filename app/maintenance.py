"""Retention only for generated job directories; Obsidian notes are preserved."""
import re
import shutil
import time

from .config import settings
from .queueing import QueueError, TERMINAL


def cleanup_uploads(store):
    if settings.UPLOAD_RETENTION_HOURS <= 0:
        return
    root = settings.UPLOAD_DIR.resolve()
    cutoff = time.time() - settings.UPLOAD_RETENTION_HOURS * 3600
    for path in root.iterdir():
        if not re.fullmatch(r"[a-f0-9]{32}", path.name) or not path.is_dir():
            continue
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            continue
        resolved = path.resolve()
        if resolved.parent != root or path.stat().st_mtime > cutoff:
            continue
        try:
            if store.get(path.name)["state"] not in TERMINAL:
                continue
        except QueueError as exc:
            if exc.code != 404:
                raise  # Redis outage is not permission to delete content.
        shutil.rmtree(resolved)

import os
import json
from config import TRACKING_FILE


def get_loaded_docs() -> list[str]:
    """Return list of loaded PDF filenames from tracking file."""
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, "r") as f:
            return json.load(f)
    return []


def add_loaded_doc(name: str) -> None:
    """Add a PDF filename to the tracking file (no duplicates)."""
    docs = get_loaded_docs()
    if name not in docs:
        docs.append(name)
        with open(TRACKING_FILE, "w") as f:
            json.dump(docs, f)


def is_already_loaded(name: str) -> bool:
    """Check if a PDF has already been indexed."""
    return name in get_loaded_docs()


def clear_loaded_docs() -> None:
    """Wipe the tracking file (used when clearing all documents)."""
    if os.path.exists(TRACKING_FILE):
        os.remove(TRACKING_FILE)
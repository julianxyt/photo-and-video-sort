"""Library paths and tunables.

Everything swipesort creates lives under ``<library>/.swipesort`` except the
quarantine folder, which sits at ``<library>/_Quarantine`` so that it is
obvious in a file browser and easy to empty by hand.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

APP_DIR_NAME = ".swipesort"
DB_NAME = "index.db"
THUMB_DIR_NAME = "thumbs"
QUARANTINE_DIR_NAME = "_Quarantine"

# Long edge of the JPEG thumbnails served to the phone. 720 keeps a swipe deck
# crisp on a 3x display without making the thumb cache bigger than it needs.
THUMB_MAX_EDGE = 720
THUMB_QUALITY = 78

# Square size the feature extractor works at. Small on purpose: features should
# describe composition and colour, not pixel-level detail.
FEATURE_IMAGE_SIZE = 128

# Two images are "near duplicates" when their dHashes differ by <= this many
# bits. 6/64 catches burst frames and re-crops without merging distinct scenes.
PHASH_HAMMING_THRESHOLD = 6

# Retrain after this many new swipes (the UI also asks explicitly).
RETRAIN_EVERY = 15

# Below this many labels the model stays silent rather than guessing.
MIN_LABELS_TO_PREDICT = 25

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765


@dataclass(frozen=True)
class Library:
    """A photo library rooted at ``root``."""

    root: Path

    @classmethod
    def resolve(cls, path: str | os.PathLike[str] | None = None) -> "Library":
        """Resolve a library from an argument, ``$SWIPESORT_LIBRARY``, or cwd."""
        raw = path or os.environ.get("SWIPESORT_LIBRARY") or Path.cwd()
        root = Path(raw).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"library root is not a directory: {root}")
        return cls(root)

    @property
    def app_dir(self) -> Path:
        return self.root / APP_DIR_NAME

    @property
    def db_path(self) -> Path:
        return self.app_dir / DB_NAME

    @property
    def thumb_dir(self) -> Path:
        return self.app_dir / THUMB_DIR_NAME

    @property
    def quarantine_dir(self) -> Path:
        return self.root / QUARANTINE_DIR_NAME

    def ensure(self) -> "Library":
        """Create the private directories, and keep them out of the scan."""
        self.thumb_dir.mkdir(parents=True, exist_ok=True)
        marker = self.app_dir / ".gitignore"
        if not marker.exists():
            marker.write_text("*\n", encoding="utf-8")
        return self

    def is_internal(self, path: Path) -> bool:
        """True for paths swipesort owns, which ingest must never index."""
        try:
            rel = path.resolve().relative_to(self.root)
        except ValueError:
            return False
        head = rel.parts[0] if rel.parts else ""
        return head in {APP_DIR_NAME, QUARANTINE_DIR_NAME}


def exiftool_path() -> str | None:
    """Locate exiftool: ``$SWIPESORT_EXIFTOOL``, then ``$PATH``, then Windows.

    The original PowerShell scripts in this repo shell out to exiftool, and the
    repo ships the 12.97 distribution, so reuse it when it is installed.
    """
    import shutil

    explicit = os.environ.get("SWIPESORT_EXIFTOOL") or os.environ.get("EXIFTOOLPATH")
    if explicit and Path(explicit).exists():
        return explicit
    found = shutil.which("exiftool") or shutil.which("exiftool.exe")
    if found:
        return found
    guess = Path(r"C:\Program Files\exiftool-12.97_64\exiftool.exe")
    return str(guess) if guess.exists() else None


def ffmpeg_path() -> str | None:
    import shutil

    return shutil.which("ffmpeg")

"""Extension and filename rules, ported from Split_To_Years_Greedy_EXIF.ps1.

The PowerShell script sorted into ``<year> <type>`` folders with a handful of
special buckets that bypassed the year lookup entirely. Those rules are kept
here verbatim in spirit, with two deliberate changes documented in the README:

* ``.nef`` moves from Photos to RAWs (it is a raw file, and grouping it with
  JPEGs made the RAWs folder useless for Nikon shooters).
* Last-write-time is never used as a date source unless explicitly asked for,
  matching the "no lastwritetime as its unreliable" commit.
"""
from __future__ import annotations

import re
from pathlib import Path

VIDEO_EXTS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".3gp", ".mts"})
PHOTO_EXTS = frozenset({".jpg", ".jpeg", ".heic", ".heif", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"})
RAW_EXTS = frozenset({".raf", ".nef", ".arw", ".cr2", ".cr3", ".dng", ".orf", ".rw2", ".pef", ".srw"})
LIVE_EXTS = frozenset({".mp"})

KNOWN_EXTS = VIDEO_EXTS | PHOTO_EXTS | RAW_EXTS | LIVE_EXTS

# Buckets that skip the year lookup, in the order the PowerShell guards ran.
BUCKET_METADATA = "Metadata"
BUCKET_LIVE = "Live Photos"
BUCKET_SCREENSHOTS = "Screenshots"
BUCKET_FACEBOOK = "Facebook"
FLAT_BUCKETS = frozenset({BUCKET_METADATA, BUCKET_LIVE, BUCKET_SCREENSHOTS, BUCKET_FACEBOOK})

# Year buckets, used as "<year> <bucket>".
BUCKET_PHOTOS = "Photos"
BUCKET_VIDEOS = "Videos"
BUCKET_RAWS = "RAWs"
BUCKET_OTHER = "Other"

UNCLASSIFIED = "Unclassified"

_SCREENSHOT_RE = re.compile(r"^screenshot", re.IGNORECASE)
_FACEBOOK_RE = re.compile(r"^(fb|received)", re.IGNORECASE)
# YYYYMMDD anywhere in the name, 2000-2099, month 01-12, day 01-39 (loose on
# purpose: only the year is used).
_FILENAME_DATE_RE = re.compile(r"(20\d{2})(0[1-9]|1[0-2])([0-3]\d)")


def media_kind(ext: str) -> str:
    """Coarse media kind: photo, video, raw, live, or other."""
    ext = ext.lower()
    if ext in LIVE_EXTS:
        return "live"
    if ext in RAW_EXTS:
        return "raw"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in PHOTO_EXTS:
        return "photo"
    return "other"


def bucket_for(filename: str, ext: str) -> str:
    """The destination bucket, applying the original guard-clause order."""
    ext = ext.lower()
    if ext not in KNOWN_EXTS:
        return BUCKET_METADATA
    if ext in LIVE_EXTS:
        return BUCKET_LIVE
    stem = Path(filename).name
    if _SCREENSHOT_RE.match(stem):
        return BUCKET_SCREENSHOTS
    if _FACEBOOK_RE.match(stem):
        return BUCKET_FACEBOOK
    kind = media_kind(ext)
    return {
        "photo": BUCKET_PHOTOS,
        "video": BUCKET_VIDEOS,
        "raw": BUCKET_RAWS,
    }.get(kind, BUCKET_OTHER)


def year_from_filename(filename: str) -> str | None:
    """Pull a year out of a YYYYMMDD run in the filename, if one is there."""
    match = _FILENAME_DATE_RE.search(Path(filename).name)
    return match.group(1) if match else None


def target_folder(bucket: str, year: str | None) -> str:
    """Folder name relative to the library root.

    Flat buckets ignore the year, exactly as the PowerShell guards did; year
    buckets become ``"2019 Photos"``, ``"Unclassified Videos"`` and so on.
    """
    if bucket in FLAT_BUCKETS:
        return bucket
    return f"{year or UNCLASSIFIED} {bucket}"


def is_media(path: Path) -> bool:
    """Whether ingest should index this file at all."""
    if path.name.startswith("."):
        return False
    return path.suffix.lower() in KNOWN_EXTS

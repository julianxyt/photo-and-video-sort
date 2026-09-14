"""Work out when a file was actually captured.

Three sources, tried in the order the PowerShell script preferred them:

1. EXIF ``DateTimeOriginal`` via Pillow (JPEG/TIFF/PNG/HEIC).
2. The QuickTime/MP4 ``mvhd`` creation time, parsed directly from the container
   so that videos do not need ffprobe installed.
3. exiftool, if it is on the machine - this is what the original scripts used,
   and it is the only thing that reads most RAW formats.

Filename pattern matching is the caller's fallback (see ``classify``); file
modification time is never used unless explicitly requested, per the
"no lastwritetime as its unreliable" commit.
"""
from __future__ import annotations

import json
import struct
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import exiftool_path

# EXIF tag ids, in preference order.
_EXIF_DATE_TAGS = (36867, 36868, 306)  # DateTimeOriginal, DateTimeDigitized, DateTime

# QuickTime epoch is 1904-01-01 UTC.
_QT_EPOCH = datetime(1904, 1, 1, tzinfo=timezone.utc)

_EXIFTOOL_FIELDS = (
    "-DateTimeOriginal",
    "-CreateDate",
    "-MediaCreateDate",
    "-SubSecDateTimeOriginal",
)


def _parse_exif_datetime(value: str) -> datetime | None:
    """EXIF stores ``YYYY:MM:DD HH:MM:SS``; tolerate a few common deviations."""
    text = str(value).strip().strip("\x00")
    if not text or text.startswith(("0000", "    ")):
        return None
    text = text.split("+")[0].split("-05:00")[0].strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y:%m:%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d"):
        try:
            return datetime.strptime(text[: len(datetime.now().strftime(fmt)) + 6].strip(), fmt)
        except ValueError:
            continue
    return None


def from_pillow(path: Path) -> datetime | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None
            for tag in _EXIF_DATE_TAGS:
                value = exif.get(tag)
                if value:
                    parsed = _parse_exif_datetime(value)
                    if parsed:
                        return parsed
    except Exception:
        return None
    return None


def from_quicktime(path: Path) -> datetime | None:
    """Walk MP4/MOV boxes to ``moov/mvhd`` and read its creation time."""
    try:
        with path.open("rb") as fh:
            return _find_mvhd(fh, end=path.stat().st_size, depth=0)
    except Exception:
        return None


def _find_mvhd(fh, end: int, depth: int) -> datetime | None:
    if depth > 4:
        return None
    while fh.tell() < end - 8:
        start = fh.tell()
        header = fh.read(8)
        if len(header) < 8:
            return None
        size, kind = struct.unpack(">I4s", header)
        if size == 1:  # 64-bit extended size
            size = struct.unpack(">Q", fh.read(8))[0]
        if size < 8 or start + size > end:
            return None
        if kind == b"mvhd":
            version = fh.read(1)[0]
            fh.read(3)  # flags
            if version == 1:
                created = struct.unpack(">Q", fh.read(8))[0]
            else:
                created = struct.unpack(">I", fh.read(4))[0]
            if created <= 0:
                return None
            stamp = _QT_EPOCH + timedelta(seconds=created)
            # Values before the format existed mean a broken writer.
            return stamp.replace(tzinfo=None) if stamp.year >= 1990 else None
        if kind in (b"moov", b"trak", b"mdia"):
            found = _find_mvhd(fh, end=start + size, depth=depth + 1)
            if found:
                return found
        fh.seek(start + size)
    return None


def from_exiftool(path: Path, tool: str | None = None) -> datetime | None:
    tool = tool or exiftool_path()
    if not tool:
        return None
    try:
        proc = subprocess.run(
            [tool, "-j", "-n", *_EXIFTOOL_FIELDS, str(path)],
            capture_output=True,
            timeout=20,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        payload = json.loads(proc.stdout.decode("utf-8", "replace"))
        if not payload:
            return None
        entry = payload[0]
        for field in ("SubSecDateTimeOriginal", "DateTimeOriginal", "CreateDate", "MediaCreateDate"):
            if entry.get(field):
                parsed = _parse_exif_datetime(str(entry[field]))
                if parsed:
                    return parsed
    except Exception:
        return None
    return None


def capture_datetime(path: Path, *, use_mtime: bool = False) -> tuple[datetime | None, str]:
    """Return ``(datetime, source)``; source is one of exif/quicktime/exiftool/mtime/none."""
    ext = path.suffix.lower()

    if ext not in {".mp4", ".mov", ".m4v", ".3gp"}:
        found = from_pillow(path)
        if found:
            return found, "exif"
    else:
        found = from_quicktime(path)
        if found:
            return found, "quicktime"

    found = from_exiftool(path)
    if found:
        return found, "exiftool"

    if use_mtime:
        try:
            return datetime.fromtimestamp(path.stat().st_mtime), "mtime"
        except OSError:
            pass
    return None, "none"

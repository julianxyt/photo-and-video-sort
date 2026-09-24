"""Work out when a file was actually captured.

Three sources, tried in the order the PowerShell script preferred them:

1. EXIF ``DateTimeOriginal`` via Pillow (JPEG/TIFF/PNG/HEIC).
2. For RAW files, the EXIF inside the camera's embedded JPEG preview, via
   rawpy - no exiftool needed, which matters because Fuji RAF filenames
   carry no date.
3. The QuickTime/MP4 ``mvhd`` creation time, parsed directly from the container
   so that videos do not need ffprobe installed.
4. exiftool, if it is on the machine - this is what the original scripts used,
   and it is the only thing that reads most RAW formats.

Filename pattern matching is the caller's fallback (see ``classify``); file
modification time is never used unless explicitly requested, per the
"no lastwritetime as its unreliable" commit.
"""
from __future__ import annotations

import json
import re
import struct
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import exiftool_path

# Where EXIF keeps its dates. The two that mean "when the shutter fired" live
# in the Exif sub-IFD, not in IFD0 - IFD0 only has DateTime, which is when the
# file was last *modified*, so an edited photo reports the day it was edited.
_EXIF_SUB_IFD = 0x8769
_TAKEN_TAGS = (36867, 36868)   # DateTimeOriginal, DateTimeDigitized (sub-IFD)
_MODIFIED_TAG = 306            # DateTime (IFD0) - last resort only

# QuickTime epoch is 1904-01-01 UTC.
_QT_EPOCH = datetime(1904, 1, 1, tzinfo=timezone.utc)

_EXIFTOOL_FIELDS = (
    "-DateTimeOriginal",
    "-CreateDate",
    "-MediaCreateDate",
    "-SubSecDateTimeOriginal",
)


_TZ_SUFFIX = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


def _parse_exif_datetime(value: str) -> datetime | None:
    """EXIF stores ``YYYY:MM:DD HH:MM:SS``; tolerate the common deviations.

    Accepts fractional seconds, a trailing timezone offset (dropped - capture
    dates are wall-clock time where the photo was taken, which is what a
    person means by "taken on"), dashes instead of colons, and a bare date.
    Blank and all-zero placeholders, which some cameras write when the clock
    was never set, come back as None.
    """
    text = str(value).strip().strip("\x00").strip()
    if not text or text.startswith(("0000", "    ")):
        return None
    text = _TZ_SUFFIX.sub("", text).strip()
    for fmt in ("%Y:%m:%d %H:%M:%S.%f", "%Y:%m:%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S", "%Y:%m:%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def exif_capture_datetime(exif) -> datetime | None:
    """Capture time from a Pillow ``Exif`` object, sub-IFD first."""
    if not exif:
        return None
    try:
        sub = exif.get_ifd(_EXIF_SUB_IFD)
    except Exception:
        sub = {}
    for tag in _TAKEN_TAGS:
        value = sub.get(tag) or exif.get(tag)
        parsed = _parse_exif_datetime(value) if value else None
        if parsed:
            return parsed
    value = exif.get(_MODIFIED_TAG)
    return _parse_exif_datetime(value) if value else None


def from_pillow(path: Path) -> datetime | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as img:
            return exif_capture_datetime(img.getexif())
    except Exception:
        return None


# LibRaw writes timestamp 0 when it found no date, which renders as a day in
# 1970 (in local time, so not reliably midnight). No RAW file predates 1990.
_EARLIEST_RAW = datetime(1990, 1, 1)


def _exif_blocks(jpeg: bytes):
    """Every EXIF (APP1) block in a JPEG, in file order."""
    i = 2
    while i + 4 <= len(jpeg) and jpeg[i] == 0xFF:
        marker = jpeg[i + 1]
        if marker == 0xDA:  # start of scan: the headers are over
            return
        length = int.from_bytes(jpeg[i + 2:i + 4], "big")
        segment = jpeg[i + 4:i + 2 + length]
        if marker == 0xE1 and segment.startswith(b"Exif\0\0"):
            yield segment
        i += 2 + length


def from_raw_preview(path: Path) -> datetime | None:
    """Read the date out of the JPEG preview embedded in a RAW file.

    Pillow cannot open most RAW containers, and Fuji names its files
    DSCF1234.RAF with no date in the name, so without this every RAF lands in
    "Unclassified RAWs" unless exiftool is installed.

    What rawpy hands back is not the camera's JPEG untouched: LibRaw prepends
    an EXIF block of its own, built from what it parsed, ahead of the camera's
    original one - and Pillow only ever reads the first. So every block is read
    here, the camera's own DateTimeOriginal is preferred wherever it appears,
    and LibRaw's "no timestamp" placeholder is rejected.
    """
    from .features import raw_preview_jpeg

    jpeg = raw_preview_jpeg(path)
    if not jpeg:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None

    taken: list[datetime] = []
    modified: list[datetime] = []
    for block in _exif_blocks(jpeg):
        exif = Image.Exif()
        try:
            exif.load(block)
            sub = exif.get_ifd(_EXIF_SUB_IFD)
        except Exception:
            continue
        for tag in _TAKEN_TAGS:
            parsed = _parse_exif_datetime(sub.get(tag) or exif.get(tag) or "")
            if parsed:
                taken.append(parsed)
        parsed = _parse_exif_datetime(exif.get(_MODIFIED_TAG) or "")
        if parsed:
            modified.append(parsed)

    for found in taken + modified:
        if found >= _EARLIEST_RAW:
            return found
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
    """Return ``(datetime, source)``.

    Source is one of exif / raw-preview / quicktime / exiftool / mtime / none.
    """
    from .classify import RAW_EXTS

    ext = path.suffix.lower()

    if ext in {".mp4", ".mov", ".m4v", ".3gp"}:
        found = from_quicktime(path)
        if found:
            return found, "quicktime"
    else:
        found = from_pillow(path)
        if found:
            return found, "exif"
        if ext in RAW_EXTS:
            found = from_raw_preview(path)
            if found:
                return found, "raw-preview"

    found = from_exiftool(path)
    if found:
        return found, "exiftool"

    if use_mtime:
        try:
            return datetime.fromtimestamp(path.stat().st_mtime), "mtime"
        except OSError:
            pass
    return None, "none"

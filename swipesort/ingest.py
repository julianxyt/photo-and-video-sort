"""Scan a library into the index: metadata, thumbnails, hashes, features.

One pass, one decode per file. The image is loaded once and used for the
thumbnail, the perceptual hash and the feature vector, because decoding is the
expensive part of ingesting forty thousand holiday photos.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from . import classify, db, features as feat, phash
from .capture_date import capture_datetime
from .config import (
    FEATURE_IMAGE_SIZE,
    Library,
    PHASH_HAMMING_THRESHOLD,
    THUMB_MAX_EDGE,
    THUMB_QUALITY,
    ffmpeg_path,
)

Progress = Callable[[str, int, int], None]


@dataclass
class IngestReport:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    skipped: int = 0
    missing: int = 0
    thumbs: int = 0
    featured: int = 0
    undecoded: int = 0
    exact_duplicates: int = 0
    duplicate_bytes: int = 0
    near_duplicate_groups: int = 0
    gaps: list["DecodeGap"] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.scanned} files scanned, {self.added} new, {self.updated} updated, "
            f"{self.thumbs} thumbnails, {self.featured} featurised, "
            f"{self.exact_duplicates} exact duplicates "
            f"({self.duplicate_bytes / 1e9:.2f} GB), "
            f"{self.near_duplicate_groups} near-duplicate groups"
        )


def walk_media(library: Library) -> Iterable[Path]:
    """Every media file under the library, skipping swipesort's own folders.

    Sorted, because os.walk yields whatever order the filesystem hands back.
    That order differs between machines and even between runs, and an unsorted
    walk makes row ids - and anything derived from them - irreproducible.
    """
    for dirpath, dirnames, filenames in os.walk(library.root):
        here = Path(dirpath)
        dirnames[:] = sorted(
            d for d in dirnames
            if not library.is_internal(here / d) and not d.startswith(".")
        )
        for name in sorted(filenames):
            path = here / name
            if classify.is_media(path):
                yield path


def scan(
    library: Library,
    *,
    use_mtime: bool = False,
    limit: int | None = None,
    extractor=None,
    progress: Progress | None = None,
    workers: int | None = None,
    dup_threshold: int = PHASH_HAMMING_THRESHOLD,
) -> IngestReport:
    library.ensure()
    conn = db.connect(library.db_path)
    extractor = extractor or feat.get_extractor("auto")
    report = IngestReport()

    known = {
        row["path"]: row
        for row in conn.execute("SELECT id, path, size_bytes, mtime, thumb, features, feat_kind FROM media")
    }
    seen: set[str] = set()

    paths = list(walk_media(library))
    if limit:
        paths = paths[:limit]
    report.scanned = len(paths)

    pending: list[tuple[int, Path, bool]] = []
    for index, path in enumerate(paths, start=1):
        key = str(path)
        seen.add(key)
        try:
            stat = path.stat()
        except OSError as exc:
            report.errors.append(f"{path}: {exc}")
            continue

        prior = known.get(key)
        unchanged = (
            prior is not None
            and prior["size_bytes"] == stat.st_size
            and prior["mtime"] == stat.st_mtime
        )
        needs_media_work = (
            not unchanged
            or prior["thumb"] is None
            or prior["features"] is None
            or prior["feat_kind"] != extractor.kind
        )

        if unchanged and not needs_media_work:
            report.skipped += 1
            continue

        record = _metadata_record(library, path, stat, use_mtime=use_mtime)
        media_id = db.upsert_media(conn, record)
        if prior is None:
            report.added += 1
        else:
            report.updated += 1
        if needs_media_work:
            pending.append((media_id, path, prior is not None and prior["thumb"] is not None))

        if index % 200 == 0:
            conn.commit()
            if progress:
                progress("indexing", index, len(paths))
    conn.commit()
    if progress:
        progress("indexing", len(paths), len(paths))

    _process_media(conn, library, extractor, pending, report, progress)

    stale = [p for p in known if p not in seen]
    if stale:
        conn.executemany("UPDATE media SET missing=1 WHERE path=?", [(p,) for p in stale])
        report.missing = len(stale)
    # A file that came back since the last scan stops being missing.
    for chunk in db.iter_chunks(seen, 500):
        conn.executemany("UPDATE media SET missing=0 WHERE path=?", [(p,) for p in chunk])
    conn.commit()

    report.exact_duplicates, report.duplicate_bytes = find_exact_duplicates(conn)
    report.near_duplicate_groups = group_near_duplicates(conn, dup_threshold)
    report.gaps = decode_gaps(conn)
    db.set_meta(conn, "last_scan", db.utcnow())
    db.set_meta(conn, "feat_kind", extractor.kind)
    conn.commit()
    conn.close()
    return report


def _metadata_record(library: Library, path: Path, stat: os.stat_result, *, use_mtime: bool) -> dict:
    ext = path.suffix.lower()
    taken, source = capture_datetime(path, use_mtime=use_mtime)
    year = None
    if taken:
        year = str(taken.year)
    else:
        year = classify.year_from_filename(path.name)
        if year:
            source = "filename"

    width = height = duration = None
    if classify.media_kind(ext) == "video":
        width, height, duration = probe_video(path)

    return {
        "path": str(path),
        "rel_path": str(path.relative_to(library.root)),
        "filename": path.name,
        "ext": ext,
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "kind": classify.media_kind(ext),
        "bucket": classify.bucket_for(path.name, ext),
        "year": year,
        "taken_at": taken.isoformat(timespec="seconds") if taken else None,
        "date_source": source,
        "width": width,
        "height": height,
        "duration_s": duration,
        "missing": 0,
    }


def probe_video(path: Path) -> tuple[int | None, int | None, float | None]:
    """Width/height/duration via ffprobe, when it is installed."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None, None, None
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height:format=duration",
             "-of", "json", str(path)],
            capture_output=True, timeout=30, check=False,
        )
        if proc.returncode != 0:
            return None, None, None
        payload = json.loads(proc.stdout or b"{}")
        stream = (payload.get("streams") or [{}])[0]
        duration = payload.get("format", {}).get("duration")
        return stream.get("width"), stream.get("height"), float(duration) if duration else None
    except Exception:
        return None, None, None


def _process_media(conn, library, extractor, pending, report: IngestReport, progress) -> None:
    """Thumbnail, hash and featurise, in a thread pool, writing from one thread."""
    if not pending:
        return
    total = len(pending)
    workers = min(8, (os.cpu_count() or 2) + 2)

    def work(job):
        media_id, path, _ = job
        return media_id, _thumb_hash_features(library, path, extractor)

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for media_id, result in pool.map(work, pending):
            done += 1
            if result is None:
                # Clear anything left from an earlier successful decode, so a
                # file replaced by an unreadable one does not keep stale pixels.
                conn.execute(
                    "UPDATE media SET thumb=NULL, phash=NULL, features=NULL, "
                    "feat_kind=NULL WHERE id=?",
                    (media_id,),
                )
                report.undecoded += 1
            else:
                thumb, value, vector, size = result
                conn.execute(
                    "UPDATE media SET thumb=?, phash=?, features=?, feat_kind=?, "
                    "width=COALESCE(?, width), height=COALESCE(?, height) WHERE id=?",
                    (thumb, value, feat.pack(vector) if vector is not None else None,
                     extractor.kind if vector is not None else None,
                     size[0] if size else None, size[1] if size else None, media_id),
                )
                if thumb:
                    report.thumbs += 1
                if vector is not None:
                    report.featured += 1
            if done % 100 == 0:
                conn.commit()
                if progress:
                    progress("thumbnails", done, total)
    conn.commit()
    if progress:
        progress("thumbnails", total, total)


def _thumb_hash_features(library: Library, path: Path, extractor):
    """Decode once; return (thumb name, phash, feature vector, (w, h)), or None."""
    from PIL import Image

    opened = feat.open_media_image(path, max_edge=THUMB_MAX_EDGE)
    if opened is None:
        return None
    image, size = opened
    try:
        thumb_name = _thumb_name(path)
        thumb_path = library.thumb_dir / thumb_name
        preview = image.copy()
        preview.thumbnail((THUMB_MAX_EDGE, THUMB_MAX_EDGE), Image.LANCZOS)
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        preview.save(thumb_path, "JPEG", quality=THUMB_QUALITY, optimize=True)

        small = np.asarray(
            image.resize((FEATURE_IMAGE_SIZE, FEATURE_IMAGE_SIZE), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        value = phash.dhash(feat.to_gray(small))
        vector = extractor.extract(small)
        return thumb_name, value, vector, size
    except Exception:
        return None
    finally:
        image.close()


def _thumb_name(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
    return f"{digest[:2]}/{digest}.jpg"


# --------------------------------------------------------------------------- #
# Files that could not be previewed
# --------------------------------------------------------------------------- #

FFMPEG_INSTALL = (
    "  Windows  winget install --id Gyan.FFmpeg\n"
    "  macOS    brew install ffmpeg\n"
    "  Linux    sudo apt install ffmpeg"
)


@dataclass
class DecodeGap:
    """A group of indexed files that have no preview and no image features."""

    kind: str       # video / heic / raw / photo / live
    label: str      # "video" / "videos", "HEIC photo" / "HEIC photos", ...
    count: int
    fix: str        # what to do about it, given what is installed right now;
                    # may run to several lines


def decode_gaps(conn: sqlite3.Connection) -> list[DecodeGap]:
    """What could not be previewed, grouped by why, with the fix for each.

    Swipes on these files still train the model, from metadata alone, so this
    is not an error - but the model learns far more when it can see the picture,
    and the fix is usually one install away.
    """
    rows = conn.execute(
        "SELECT kind, ext, COUNT(*) AS n FROM media "
        "WHERE features IS NULL AND missing=0 AND exact_dup_of IS NULL "
        "GROUP BY kind, ext"
    ).fetchall()
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["kind"] == "video":
            counts["video"] += row["n"]
        elif row["ext"] in (".heic", ".heif"):
            counts["heic"] += row["n"]
        elif row["kind"] == "raw":
            counts["raw"] += row["n"]
        elif row["kind"] == "live":
            counts["live"] += row["n"]
        else:
            counts["photo"] += row["n"]

    fixes = {
        "video": ("video",
                  f"install ffmpeg, then re-run ingest:\n{FFMPEG_INSTALL}"
                  if not ffmpeg_path() else
                  "ffmpeg could not read them - corrupt files, or a codec it lacks"),
        "heic": ("HEIC photo",
                 "pip install pillow-heif, then re-run ingest"
                 if not feat.HEIF_AVAILABLE else
                 "pillow-heif could not read them - likely corrupt"),
        "raw": ("RAW file",
                "pip install rawpy, then re-run ingest"
                if not feat.rawpy_available() else
                "rawpy could not decode them - a camera newer than its LibRaw, or corrupt"),
        "live": ("Live/Motion Photo clip", "not previewed yet - still indexed and sortable"),
        "photo": ("photo", "could not be decoded - likely corrupt or truncated"),
    }
    order = ["video", "heic", "raw", "photo", "live"]
    gaps = []
    for kind in order:
        count = counts.get(kind, 0)
        if count:
            label, fix = fixes[kind]
            gaps.append(DecodeGap(kind, label if count == 1 else label + "s", count, fix))
    return gaps


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def survivor_rank(rel_path: str) -> tuple[int, int, str]:
    """Sort key deciding which of several byte-identical files is kept.

    Shallowest path first, then shortest, then alphabetical.

    Which one survives is arbitrary in the sense that the bytes are the same
    either way - but it must be *decided*, not left to chance. Keeping whichever
    file was ingested first meant keeping whichever one os.walk reached first,
    so the survivor depended on the order the filesystem happened to return
    directory entries in, and a rescan could pick a different one.
    """
    return (len(Path(rel_path).parts), len(rel_path), rel_path)


def find_exact_duplicates(conn: sqlite3.Connection) -> tuple[int, int]:
    """Hash only files that share a byte size with another file, then link them.

    The original Remove-DuplicateFiles matched on name alone, which deletes two
    different photos that happen to share a name. Matching on content means
    nothing unique is ever lost; ``survivor_rank`` decides which copy stays.
    """
    groups = conn.execute(
        "SELECT size_bytes FROM media WHERE missing=0 GROUP BY size_bytes HAVING COUNT(*) > 1"
    ).fetchall()
    total = 0
    total_bytes = 0
    for (size,) in ((g["size_bytes"],) for g in groups):
        rows = sorted(
            conn.execute(
                "SELECT id, path, rel_path, sha256 FROM media "
                "WHERE size_bytes=? AND missing=0",
                (size,),
            ).fetchall(),
            key=lambda row: survivor_rank(row["rel_path"]),
        )
        digests: dict[str, int] = {}
        for row in rows:
            digest = row["sha256"]
            if not digest:
                try:
                    digest = sha256_file(Path(row["path"]))
                except OSError:
                    continue
                conn.execute("UPDATE media SET sha256=? WHERE id=?", (digest, row["id"]))
            first = digests.get(digest)
            if first is None:
                digests[digest] = row["id"]
                conn.execute("UPDATE media SET exact_dup_of=NULL WHERE id=?", (row["id"],))
            else:
                conn.execute("UPDATE media SET exact_dup_of=? WHERE id=?", (first, row["id"]))
                total += 1
                total_bytes += size
    conn.commit()
    return total, total_bytes


def group_near_duplicates(conn: sqlite3.Connection, threshold: int = PHASH_HAMMING_THRESHOLD) -> int:
    """Cluster visually similar images and store a dense group id per row."""
    rows = conn.execute(
        "SELECT id, phash FROM media WHERE phash IS NOT NULL AND missing=0 AND exact_dup_of IS NULL"
    ).fetchall()
    mapping = phash.group_near_duplicates([(r["id"], r["phash"]) for r in rows], threshold)
    packed = phash.pack_groups(mapping)
    conn.execute("UPDATE media SET dup_group=NULL")
    conn.executemany(
        "UPDATE media SET dup_group=? WHERE id=?",
        [(gid, mid) for mid, gid in packed.items()],
    )
    conn.commit()
    return len(set(packed.values()))

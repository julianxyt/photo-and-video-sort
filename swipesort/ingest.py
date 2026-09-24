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
    exact_duplicates: int = 0
    duplicate_bytes: int = 0
    near_duplicate_groups: int = 0
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
    """Every media file under the library, skipping swipesort's own folders."""
    for dirpath, dirnames, filenames in os.walk(library.root):
        here = Path(dirpath)
        dirnames[:] = [d for d in dirnames if not library.is_internal(here / d) and not d.startswith(".")]
        for name in filenames:
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
                report.errors.append(f"could not decode: {media_id}")
            else:
                thumb, value, vector, size = result
                conn.execute(
                    "UPDATE media SET thumb=?, phash=?, features=?, feat_kind=?, "
                    "width=COALESCE(width,?), height=COALESCE(height,?) WHERE id=?",
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
    """Decode once; return (thumb name, phash, feature vector, (w, h))."""
    from PIL import Image

    thumb_name = _thumb_name(path)
    thumb_path = library.thumb_dir / thumb_name
    size = None

    try:
        if path.suffix.lower() in classify.VIDEO_EXTS:
            image = feat._video_frame(path)
            if image is None:
                return thumb_name if thumb_path.exists() else None, None, None, None
        else:
            image = Image.open(path)
            image.draft("RGB", (THUMB_MAX_EDGE, THUMB_MAX_EDGE))
            image = image.convert("RGB")
        size = image.size

        preview = image.copy()
        preview.thumbnail((THUMB_MAX_EDGE, THUMB_MAX_EDGE), Image.LANCZOS)
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        preview.save(thumb_path, "JPEG", quality=THUMB_QUALITY, optimize=True)

        small = np.asarray(
            image.resize((FEATURE_IMAGE_SIZE, FEATURE_IMAGE_SIZE), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        gray = feat.to_gray(small)
        value = phash.dhash(gray)
        vector = extractor.extract(small)
        image.close()
        return thumb_name, value, vector, size
    except Exception:
        return None


def _thumb_name(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
    return f"{digest[:2]}/{digest}.jpg"


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def find_exact_duplicates(conn: sqlite3.Connection) -> tuple[int, int]:
    """Hash only files that share a byte size with another file, then link them.

    The original Remove-DuplicateFiles matched on name alone, which deletes two
    different photos that happen to share a name. Matching on content means the
    survivor is chosen by path order and nothing unique is ever lost.
    """
    groups = conn.execute(
        "SELECT size_bytes FROM media WHERE missing=0 GROUP BY size_bytes HAVING COUNT(*) > 1"
    ).fetchall()
    total = 0
    total_bytes = 0
    for (size,) in ((g["size_bytes"],) for g in groups):
        rows = conn.execute(
            "SELECT id, path, sha256 FROM media WHERE size_bytes=? AND missing=0 ORDER BY id",
            (size,),
        ).fetchall()
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

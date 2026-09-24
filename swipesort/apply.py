"""Turn swipes into file moves - reversibly.

Nothing is ever deleted. Dropped files are moved into
``<library>/_Quarantine/<batch>/`` keeping their original relative path, and
every move is logged so ``swipesort undo`` can put the library back exactly as
it was. Emptying the quarantine is a separate, explicit command.

Keeps are filed into ``<year> <bucket>`` folders, the same layout
Split_To_Years_Greedy_EXIF.ps1 produced, so an existing sorted library stays
consistent. Name collisions are resolved by content: identical files collapse
into one, different files both survive under distinct names.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import db
from .classify import target_folder
from .config import Library
from .ingest import sha256_file

KEEP_ACTIONS = ("keep", "love")


@dataclass
class ApplyReport:
    batch: str
    quarantined: int = 0
    quarantined_bytes: int = 0
    sorted_: int = 0
    collapsed: int = 0
    unchanged: int = 0
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)
    moves: list[tuple[str, str, str]] = field(default_factory=list)

    def summary(self) -> str:
        verb = "would move" if self.dry_run else "moved"
        return (
            f"batch {self.batch}: {verb} {self.quarantined} file(s) to quarantine "
            f"({self.quarantined_bytes / 1e9:.2f} GB), filed {self.sorted_}, "
            f"collapsed {self.collapsed} duplicate(s), left {self.unchanged} in place"
        )


def new_batch_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def apply_decisions(
    library: Library,
    *,
    quarantine_drops: bool = True,
    sort_keeps: bool = True,
    include_exact_duplicates: bool = True,
    prune_empty: bool = False,
    dry_run: bool = False,
) -> ApplyReport:
    conn = db.connect(library.db_path)
    batch = new_batch_id()
    report = ApplyReport(batch=batch, dry_run=dry_run)

    if quarantine_drops:
        rows = conn.execute(
            "SELECT m.* FROM media m JOIN verdicts v ON v.media_id=m.id "
            "WHERE v.action='drop' AND m.missing=0"
        ).fetchall()
        for row in rows:
            _quarantine(conn, library, row, batch, report, reason="dropped", dry_run=dry_run)

    if include_exact_duplicates:
        rows = conn.execute(
            "SELECT m.* FROM media m LEFT JOIN verdicts v ON v.media_id=m.id "
            "WHERE m.exact_dup_of IS NOT NULL AND m.missing=0 "
            "AND (v.action IS NULL OR v.action <> 'keep')"
        ).fetchall()
        for row in rows:
            _quarantine(conn, library, row, batch, report, reason="exact-duplicate", dry_run=dry_run)
            report.collapsed += 1

    if sort_keeps:
        rows = conn.execute(
            "SELECT m.* FROM media m JOIN verdicts v ON v.media_id=m.id "
            f"WHERE v.action IN ({','.join('?' * len(KEEP_ACTIONS))}) AND m.missing=0 "
            "AND m.exact_dup_of IS NULL",
            KEEP_ACTIONS,
        ).fetchall()
        for row in rows:
            _file_keeper(conn, library, row, batch, report, dry_run=dry_run)

    if not dry_run:
        _write_manifest(library, batch, report)
        conn.commit()
        if prune_empty:
            # Only folders the moves themselves emptied, never the library root.
            for folder in {Path(src).parent for _, src, _ in report.moves}:
                _prune_upwards(folder, stop=library.root)
    conn.close()
    return report


def _quarantine(conn, library: Library, row, batch: str, report: ApplyReport, *, reason: str,
                dry_run: bool) -> None:
    src = Path(row["path"])
    if not src.exists():
        conn.execute("UPDATE media SET missing=1 WHERE id=?", (row["id"],))
        return
    dst = _unique(library.quarantine_dir / batch / reason / row["rel_path"])
    report.moves.append((reason, str(src), str(dst)))
    report.quarantined += 1
    report.quarantined_bytes += row["size_bytes"] or 0
    if dry_run:
        return
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    except OSError as exc:
        report.errors.append(f"{src}: {exc}")
        report.quarantined -= 1
        report.quarantined_bytes -= row["size_bytes"] or 0
        return
    _log_move(conn, batch, row["id"], f"quarantine:{reason}", src, dst)
    conn.execute("UPDATE media SET path=?, rel_path=? WHERE id=?",
                 (str(dst), str(dst.relative_to(library.root)), row["id"]))


def _file_keeper(conn, library: Library, row, batch: str, report: ApplyReport, *, dry_run: bool) -> None:
    src = Path(row["path"])
    if not src.exists():
        conn.execute("UPDATE media SET missing=1 WHERE id=?", (row["id"],))
        return
    folder = library.root / target_folder(row["bucket"], row["year"])
    dst = folder / src.name

    if dst.resolve() == src.resolve():
        report.unchanged += 1
        return

    if dst.exists():
        # Same name at the destination: if the bytes match it is the same photo
        # arriving twice, so collapse it; otherwise both survive.
        try:
            same = sha256_file(src) == sha256_file(dst)
        except OSError as exc:
            report.errors.append(f"{src}: {exc}")
            return
        if same:
            _quarantine(conn, library, row, batch, report, reason="exact-duplicate", dry_run=dry_run)
            report.collapsed += 1
            return
        dst = _unique(dst)

    report.moves.append(("keep", str(src), str(dst)))
    report.sorted_ += 1
    if dry_run:
        return
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    except OSError as exc:
        report.errors.append(f"{src}: {exc}")
        report.sorted_ -= 1
        return
    _log_move(conn, batch, row["id"], "sort", src, dst)
    conn.execute("UPDATE media SET path=?, rel_path=? WHERE id=?",
                 (str(dst), str(dst.relative_to(library.root)), row["id"]))


def _unique(path: Path) -> Path:
    """First free name of the form ``stem.ext``, ``stem (2).ext``, ...

    A counter rather than the original script's random suffix, so a re-run
    produces the same names and the folder stays readable.
    """
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for n in range(2, 10_000):
        candidate = path.with_name(f"{stem} ({n}){suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find a free filename near {path}")


def _log_move(conn: sqlite3.Connection, batch: str, media_id: int, op: str, src: Path, dst: Path) -> None:
    conn.execute(
        "INSERT INTO moves(batch, media_id, op, src, dst, created_at) VALUES(?,?,?,?,?,?)",
        (batch, media_id, op, str(src), str(dst), db.utcnow()),
    )


def _write_manifest(library: Library, batch: str, report: ApplyReport) -> None:
    if not report.moves:
        return
    folder = library.quarantine_dir / batch
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "batch": batch,
                "created_at": db.utcnow(),
                "summary": report.summary(),
                "moves": [{"op": op, "src": src, "dst": dst} for op, src, dst in report.moves],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def undo_batch(library: Library, batch: str | None = None) -> tuple[int, list[str]]:
    """Move every file in ``batch`` (default: the most recent) back where it was."""
    conn = db.connect(library.db_path)
    if batch is None:
        row = conn.execute(
            "SELECT batch FROM moves WHERE undone_at IS NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            conn.close()
            return 0, ["nothing to undo"]
        batch = row["batch"]

    moves = conn.execute(
        "SELECT * FROM moves WHERE batch=? AND undone_at IS NULL ORDER BY id DESC", (batch,)
    ).fetchall()
    restored, errors = 0, []
    for move in moves:
        src, dst = Path(move["src"]), Path(move["dst"])
        if not dst.exists():
            errors.append(f"missing, cannot restore: {dst}")
            continue
        try:
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), str(_unique(src)))
        except OSError as exc:
            errors.append(f"{dst}: {exc}")
            continue
        conn.execute("UPDATE moves SET undone_at=? WHERE id=?", (db.utcnow(), move["id"]))
        conn.execute("UPDATE media SET path=?, rel_path=? WHERE id=?",
                     (str(src), str(src.relative_to(library.root)), move["media_id"]))
        restored += 1
    conn.commit()
    conn.close()
    _prune_empty(library.quarantine_dir)
    return restored, errors


def empty_quarantine(library: Library, batch: str | None = None) -> tuple[int, int]:
    """Permanently delete quarantined files. The only destructive command here."""
    target = library.quarantine_dir / batch if batch else library.quarantine_dir
    if not target.exists():
        return 0, 0
    count = bytes_freed = 0
    for path in target.rglob("*"):
        if path.is_file():
            count += 1
            bytes_freed += path.stat().st_size
    shutil.rmtree(target)
    conn = db.connect(library.db_path)
    conn.execute(
        "UPDATE media SET missing=1 WHERE path LIKE ?", (f"{target}%",)
    )
    conn.commit()
    conn.close()
    return count, bytes_freed


def _prune_upwards(folder: Path, *, stop: Path) -> None:
    """Remove ``folder`` and its now-empty parents, stopping short of ``stop``."""
    current = folder
    while current != stop and stop in current.parents:
        try:
            if any(current.iterdir()):
                return
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _prune_empty(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()

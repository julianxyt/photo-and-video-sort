"""Decide what to show next.

Four orderings, because "what should I look at next" has different answers
depending on why you opened the app:

``learn``   the model is least sure - these swipes teach it the most
``clean``   likely rubbish, biggest first - fastest route to free space
``keepers`` likely favourites first - for picking the ones worth editing
``backlog`` oldest first - the straightforward chronological grind

Whatever the ordering, near-duplicates are always kept adjacent so that the
seven near-identical frames of the same doorway arrive together and you can
keep the best one while it is still on screen.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Iterable

from . import db, model as model_mod

MODES = ("learn", "clean", "keepers", "backlog")


def build(
    conn: sqlite3.Connection,
    *,
    mode: str = "learn",
    limit: int = 20,
    offset: int = 0,
    include_later: bool = False,
    year: str | None = None,
    bucket: str | None = None,
) -> list[dict[str, Any]]:
    """Return the next ``limit`` items to triage, best-first for ``mode``."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    rows = db.undecided_rows(conn, include_later=include_later)
    if year:
        rows = [r for r in rows if (r["year"] or "Unclassified") == year]
    if bucket:
        rows = [r for r in rows if r["bucket"] == bucket]
    if not rows:
        return []

    group_sizes = model_mod.dup_group_sizes(conn)
    has_model = any(r["score"] is not None for r in rows)
    max_size = max((r["size_bytes"] or 0) for r in rows) or 1

    scored = [
        (_priority(r, mode, has_model, max_size), r) for r in rows
    ]
    ordered = _cluster(scored)
    window = ordered[offset: offset + limit]
    return [_present(r, group_sizes, rank) for rank, r in enumerate(window, start=offset + 1)]


def _priority(row: sqlite3.Row, mode: str, has_model: bool, max_size: int) -> float:
    """Lower sorts earlier."""
    score = row["score"]
    size_ratio = (row["size_bytes"] or 0) / max_size

    if mode == "backlog":
        return _chronological_key(row)

    if not has_model or score is None:
        # Cold start: no model yet, so fall back to signals that are useful on
        # their own and that also happen to produce informative first labels.
        if mode == "clean":
            return -size_ratio
        if mode == "keepers":
            return -(row["width"] or 0) * (row["height"] or 0) / 1e8
        # learn: spread across the library rather than marching through one day
        return (hash((row["id"], row["year"] or "")) % 10_000) / 10_000.0

    if mode == "learn":
        # Closest to the decision boundary first, with a nudge towards big
        # files so that an uncertain 4 GB video outranks an uncertain 200 KB one.
        return abs(score - 0.5) - 0.05 * size_ratio
    if mode == "clean":
        # Two tiers, deliberately. Within "the model thinks this is rubbish",
        # order by how many bytes binning it actually reclaims; everything the
        # model likes waits behind all of it. Ranking purely by expected bytes
        # would surface a big file the model half-likes ahead of a small file
        # it is certain about, which is not what "show me the rubbish" means.
        tier = 0.0 if score < 0.5 else 10.0
        return tier - (1.0 - score) * (0.3 + size_ratio)
    if mode == "keepers":
        return -score
    return 0.0


def _chronological_key(row: sqlite3.Row) -> float:
    taken = row["taken_at"]
    if not taken:
        return float("inf")
    try:
        from datetime import datetime

        return datetime.fromisoformat(str(taken)).timestamp()
    except ValueError:
        return float("inf")


def _cluster(scored: Iterable[tuple[float, sqlite3.Row]]) -> list[sqlite3.Row]:
    """Sort by priority, but pull each near-duplicate group together."""
    items = list(scored)
    best_for_group: dict[int, float] = {}
    for priority, row in items:
        gid = row["dup_group"]
        if gid is not None:
            best_for_group[gid] = min(best_for_group.get(gid, priority), priority)

    def key(entry: tuple[float, sqlite3.Row]):
        priority, row = entry
        gid = row["dup_group"]
        anchor = best_for_group.get(gid, priority) if gid is not None else priority
        # A group travels together, led by its own best-ranked member, then by
        # size so the full-size frame arrives before the downscaled copy.
        return (anchor, gid if gid is not None else -1, priority,
                -(row["size_bytes"] or 0), row["id"])

    return [row for _, row in sorted(items, key=key)]


def _present(row: sqlite3.Row, group_sizes: dict[int, int], rank: int) -> dict[str, Any]:
    gid = row["dup_group"]
    return {
        "id": row["id"],
        "rank": rank,
        "filename": row["filename"],
        "rel_path": row["rel_path"],
        "kind": row["kind"],
        "bucket": row["bucket"],
        "ext": row["ext"],
        "year": row["year"] or "Unclassified",
        "taken_at": row["taken_at"],
        "date_source": row["date_source"],
        "size_bytes": row["size_bytes"],
        "width": row["width"],
        "height": row["height"],
        "duration_s": row["duration_s"],
        "score": row["score"],
        "dup_group": gid,
        "dup_group_size": group_sizes.get(gid, 1) if gid is not None else 1,
        "deferred": row["action"] == "later" if "action" in row.keys() else False,
        "has_thumb": row["thumb"] is not None,
        "has_features": row["features"] is not None,
        "thumb_url": f"/thumb/{row['id']}",
        "media_url": f"/media/{row['id']}",
        "target_folder": _target(row),
    }


def _target(row: sqlite3.Row) -> str:
    from .classify import target_folder

    return target_folder(row["bucket"], row["year"])

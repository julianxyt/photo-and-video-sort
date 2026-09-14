"""HTTP API and the phone UI.

Runs on your machine, serves to your phone over the local network. There is no
authentication and no upload path: this is a LAN tool that hands out your photo
library, so bind it to a network you trust and do not port-forward it.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import apply as apply_mod, db, model as model_mod, queue as queue_mod
from .config import Library, MIN_LABELS_TO_PREDICT, RETRAIN_EVERY

WEB_DIR = Path(__file__).parent / "web"


class Decision(BaseModel):
    id: int
    action: str = Field(pattern="^(keep|love|drop|later)$")
    latency_ms: int | None = None
    drop_rest_of_group: bool = False


class ApplyRequest(BaseModel):
    confirm: bool = False
    sort_keeps: bool = True
    quarantine_drops: bool = True
    include_exact_duplicates: bool = True


def create_app(library: Library) -> FastAPI:
    library.ensure()
    app = FastAPI(title="swipesort", docs_url="/api/docs", redoc_url=None)
    app.state.library = library
    app.state.since_train = 0

    def conn() -> sqlite3.Connection:
        # SQLite connections are not shareable between threads, and Starlette
        # runs sync endpoints in a threadpool, so open per request.
        return db.connect(library.db_path)

    # -- UI ---------------------------------------------------------------- #

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        page = WEB_DIR / "index.html"
        if not page.exists():
            raise HTTPException(500, "web assets missing")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    # -- state and queue --------------------------------------------------- #

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        c = conn()
        try:
            report = db.get_meta(c, "model_report")
            years = [
                r["year"] or "Unclassified"
                for r in c.execute(
                    "SELECT DISTINCT COALESCE(year,'Unclassified') AS year FROM media "
                    "WHERE missing=0 ORDER BY year DESC"
                )
            ]
            buckets = [
                r["bucket"]
                for r in c.execute(
                    "SELECT bucket, COUNT(*) n FROM media WHERE missing=0 "
                    "GROUP BY bucket ORDER BY n DESC"
                )
            ]
            return {
                "library": str(library.root),
                "counts": db.counts(c),
                "model": report,
                "min_labels": MIN_LABELS_TO_PREDICT,
                "retrain_every": RETRAIN_EVERY,
                "modes": list(queue_mod.MODES),
                "years": years,
                "buckets": buckets,
                "last_scan": db.get_meta(c, "last_scan"),
                "feat_kind": db.get_meta(c, "feat_kind"),
            }
        finally:
            c.close()

    @app.get("/api/queue")
    def get_queue(
        mode: str = "learn",
        limit: int = 20,
        offset: int = 0,
        include_later: bool = False,
        year: str | None = None,
        bucket: str | None = None,
    ) -> dict[str, Any]:
        if mode not in queue_mod.MODES:
            raise HTTPException(400, f"unknown mode {mode!r}")
        c = conn()
        try:
            items = queue_mod.build(
                c, mode=mode, limit=min(limit, 100), offset=offset,
                include_later=include_later, year=year, bucket=bucket,
            )
            return {"mode": mode, "items": items, "counts": db.counts(c)}
        finally:
            c.close()

    # -- decisions --------------------------------------------------------- #

    @app.post("/api/decide")
    def decide(decision: Decision) -> dict[str, Any]:
        c = conn()
        try:
            row = c.execute("SELECT * FROM media WHERE id=?", (decision.id,)).fetchone()
            if row is None:
                raise HTTPException(404, f"no media with id {decision.id}")
            db.record_decision(
                c, decision.id, decision.action, latency_ms=decision.latency_ms
            )
            also = 0
            if decision.drop_rest_of_group and row["dup_group"] is not None:
                siblings = c.execute(
                    "SELECT m.id FROM media m LEFT JOIN verdicts v ON v.media_id=m.id "
                    "WHERE m.dup_group=? AND m.id<>? AND m.missing=0 "
                    "AND (v.action IS NULL OR v.action='later')",
                    (row["dup_group"], decision.id),
                ).fetchall()
                for sibling in siblings:
                    db.record_decision(c, sibling["id"], "drop", source="group")
                also = len(siblings)

            app.state.since_train += 1 + also
            retrained = None
            if app.state.since_train >= RETRAIN_EVERY:
                retrained = _retrain(c)
                app.state.since_train = 0
            return {
                "ok": True,
                "also_dropped": also,
                "counts": db.counts(c),
                "model": retrained,
            }
        finally:
            c.close()

    @app.post("/api/undo")
    def undo() -> dict[str, Any]:
        c = conn()
        try:
            removed = db.undo_last_decision(c)
            if removed is None:
                return {"ok": False, "reason": "nothing to undo"}
            row = c.execute("SELECT * FROM media WHERE id=?", (removed["media_id"],)).fetchone()
            group_sizes = model_mod.dup_group_sizes(c)
            item = queue_mod._present(
                c.execute(
                    "SELECT m.*, NULL AS action FROM media m WHERE m.id=?", (removed["media_id"],)
                ).fetchone(),
                group_sizes,
                0,
            ) if row else None
            return {
                "ok": True,
                "undone": {"media_id": removed["media_id"], "action": removed["action"]},
                "item": item,
                "counts": db.counts(c),
            }
        finally:
            c.close()

    @app.post("/api/train")
    def train() -> dict[str, Any]:
        c = conn()
        try:
            result = _retrain(c)
            app.state.since_train = 0
            if result is None:
                counts = db.counts(c)
                needed = MIN_LABELS_TO_PREDICT - (counts["keep"] + counts["love"] + counts["drop"])
                return {"ok": False, "reason": f"need about {max(needed, 1)} more swipes"}
            return {"ok": True, "model": result}
        finally:
            c.close()

    def _retrain(c: sqlite3.Connection) -> dict[str, Any] | None:
        report = model_mod.train(c)
        if report is None:
            return None
        model_mod.score_all(c)
        return report.as_dict()

    # -- files ------------------------------------------------------------- #

    @app.get("/thumb/{media_id}")
    def thumb(media_id: int) -> FileResponse:
        c = conn()
        try:
            row = c.execute("SELECT thumb FROM media WHERE id=?", (media_id,)).fetchone()
        finally:
            c.close()
        if row is None or not row["thumb"]:
            raise HTTPException(404, "no thumbnail")
        path = library.thumb_dir / row["thumb"]
        if not path.exists():
            raise HTTPException(404, "thumbnail missing on disk")
        return FileResponse(path, media_type="image/jpeg",
                            headers={"Cache-Control": "public, max-age=604800"})

    @app.get("/media/{media_id}")
    def media(media_id: int, request: Request) -> FileResponse:
        c = conn()
        try:
            row = c.execute("SELECT path, ext, kind FROM media WHERE id=?", (media_id,)).fetchone()
        finally:
            c.close()
        if row is None:
            raise HTTPException(404, "unknown media")
        path = Path(row["path"])
        if not path.exists():
            raise HTTPException(404, "file missing on disk")
        if not _within(path, library.root):
            raise HTTPException(403, "outside the library")
        return FileResponse(path, filename=path.name)

    # -- apply ------------------------------------------------------------- #

    @app.post("/api/apply")
    def run_apply(request: ApplyRequest) -> JSONResponse:
        report = apply_mod.apply_decisions(
            library,
            quarantine_drops=request.quarantine_drops,
            sort_keeps=request.sort_keeps,
            include_exact_duplicates=request.include_exact_duplicates,
            dry_run=not request.confirm,
        )
        return JSONResponse({
            "dry_run": report.dry_run,
            "batch": report.batch,
            "quarantined": report.quarantined,
            "quarantined_bytes": report.quarantined_bytes,
            "sorted": report.sorted_,
            "collapsed": report.collapsed,
            "unchanged": report.unchanged,
            "errors": report.errors[:20],
            "summary": report.summary(),
        })

    return app


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False

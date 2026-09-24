"""Command line entry point: ``python -m swipesort <command>``."""
from __future__ import annotations

import argparse
import socket
import sys

from . import apply as apply_mod, db, features as feat, ingest, model as model_mod, queue as queue_mod
from .config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    Library,
    PHASH_HAMMING_THRESHOLD,
    exiftool_path,
    ffmpeg_path,
)


def _library(args) -> Library:
    return Library.resolve(args.library)


def _gb(n: int | float) -> str:
    return f"{(n or 0) / 1e9:.2f} GB"


def _progress(stage: str, done: int, total: int) -> None:
    pct = 100 * done / max(total, 1)
    end = "\n" if done >= total else "\r"
    print(f"  {stage}: {done}/{total} ({pct:.0f}%)", end=end, flush=True)


# --------------------------------------------------------------------------- #

def cmd_ingest(args) -> int:
    library = _library(args)
    print(f"library: {library.root}")
    backend = feat.get_extractor(args.backend)
    print(f"features: {backend.kind}")
    if not ffmpeg_path():
        print("  note: ffmpeg not found - videos get metadata only, no thumbnail or features")
    if not exiftool_path():
        print("  note: exiftool not found - RAW capture dates fall back to the filename")
        print("        (the bundled zip unpacks as 'exiftool(-k).exe'; rename it to "
              "'exiftool.exe' -")
        print("         the (-k) build waits for a keypress before exiting and would hang here)")
    report = ingest.scan(
        library,
        use_mtime=args.use_mtime,
        limit=args.limit,
        extractor=backend,
        dup_threshold=args.dup_threshold,
        progress=None if args.quiet else _progress,
    )
    print(report.summary())
    if report.errors:
        print(f"{len(report.errors)} error(s); first few:")
        for line in report.errors[:5]:
            print(f"  {line}")
    return 0


def cmd_stats(args) -> int:
    library = _library(args)
    conn = db.connect(library.db_path)
    counts = db.counts(conn)
    print(f"library: {library.root}")
    print(f"  indexed          {counts['total']:>8,}  ({_gb(counts['total_bytes'])})")
    print(f"  decided          {counts['decided']:>8,}")
    print(f"    keep/love      {counts['keep'] + counts['love']:>8,}")
    print(f"    drop           {counts['drop']:>8,}  ({_gb(counts['drop_bytes'])} reclaimable)")
    print(f"    later          {counts['later']:>8,}")
    print(f"  remaining        {counts['remaining']:>8,}")
    print(f"  exact duplicates {counts['exact_duplicates']:>8,}  ({_gb(counts['exact_duplicate_bytes'])})")

    report = db.get_meta(conn, "model_report")
    if report:
        acc = report.get("holdout_accuracy")
        auc = report.get("holdout_auc")
        base = report.get("baseline_accuracy")
        print("\nmodel")
        print(f"  trained on       {report['n_labels']:>8,} labels ({report['n_positive']} keep)")
        print(f"  features         {report['n_features']:>8,} ({report['feat_kind']})")
        if acc is not None:
            gain = f", baseline {base:.0%}" if base is not None else ""
            print(f"  holdout accuracy {acc:>8.0%}{gain}")
        if auc is not None:
            print(f"  holdout AUC      {auc:>8.2f}")
    else:
        print("\nmodel: not trained yet")
    conn.close()
    return 0


def cmd_train(args) -> int:
    library = _library(args)
    conn = db.connect(library.db_path)
    report = model_mod.train(conn, l2=args.l2)
    if report is None:
        print("not enough labels yet - swipe some more, then try again")
        conn.close()
        return 1
    scored = model_mod.score_all(conn)
    print(f"trained on {report.n_labels} labels ({report.n_positive} keep), "
          f"{report.n_features} features [{report.feat_kind}]")
    if report.holdout_accuracy is not None:
        baseline = f" (baseline {report.baseline_accuracy:.0%})" if report.baseline_accuracy else ""
        print(f"holdout accuracy {report.holdout_accuracy:.0%}{baseline}, "
              f"AUC {report.holdout_auc:.2f}" if report.holdout_auc else "")
    print(f"scored {scored} items")
    conn.close()
    return 0


def cmd_queue(args) -> int:
    library = _library(args)
    conn = db.connect(library.db_path)
    items = queue_mod.build(conn, mode=args.mode, limit=args.limit)
    if not items:
        print("queue is empty")
    for item in items:
        score = "  --" if item["score"] is None else f"{item['score']:.2f}"
        dup = f" dup×{item['dup_group_size']}" if item["dup_group_size"] > 1 else ""
        print(f"{score}  {item['size_bytes'] / 1e6:7.1f} MB  {item['year']:>12}  "
              f"{item['rel_path']}{dup}")
    conn.close()
    return 0


def cmd_apply(args) -> int:
    library = _library(args)
    report = apply_mod.apply_decisions(
        library,
        quarantine_drops=not args.no_quarantine,
        sort_keeps=not args.no_sort,
        include_exact_duplicates=not args.keep_duplicates,
        prune_empty=args.prune_empty,
        dry_run=not args.confirm,
    )
    print(report.summary())
    if report.dry_run:
        for op, src, dst in report.moves[:15]:
            print(f"  [{op}] {src} -> {dst}")
        if len(report.moves) > 15:
            print(f"  ... and {len(report.moves) - 15} more")
        print("\nnothing has moved. re-run with --confirm to apply.")
    else:
        print(f"undo with: python -m swipesort undo --batch {report.batch}")
    for line in report.errors[:10]:
        print(f"  error: {line}")
    return 0


def cmd_undo(args) -> int:
    library = _library(args)
    restored, errors = apply_mod.undo_batch(library, args.batch)
    print(f"restored {restored} file(s)")
    for line in errors[:10]:
        print(f"  {line}")
    return 0


def cmd_empty(args) -> int:
    library = _library(args)
    target = library.quarantine_dir / args.batch if args.batch else library.quarantine_dir
    if not args.confirm:
        print(f"would permanently delete: {target}")
        print("this cannot be undone. re-run with --confirm.")
        return 0
    count, freed = apply_mod.empty_quarantine(library, args.batch)
    print(f"deleted {count} file(s), freed {_gb(freed)}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .server import create_app

    library = _library(args)
    app = create_app(library)
    print(f"library: {library.root}")
    for url in _lan_urls(args.port):
        print(f"  open on your phone: {url}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _lan_urls(port: int) -> list[str]:
    urls = [f"http://localhost:{port}"]
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        urls.append(f"http://{probe.getsockname()[0]}:{port}")
        probe.close()
    except OSError:
        pass
    return urls


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swipesort",
        description="Swipe-to-triage a photo and video library, and learn what you keep.",
    )
    parser.add_argument("--library", "-L", default=None,
                        help="library root (default: $SWIPESORT_LIBRARY or the current directory)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="scan the library: metadata, thumbnails, hashes, features")
    p.add_argument("--backend", choices=["auto", "classic", "clip"], default="auto")
    p.add_argument("--limit", type=int, default=None, help="only process the first N files")
    p.add_argument("--use-mtime", action="store_true",
                   help="fall back to file modification time for undated files (unreliable)")
    p.add_argument("--dup-threshold", type=int, default=PHASH_HAMMING_THRESHOLD,
                   help="bit distance for near-duplicate grouping; lower is stricter "
                        f"(default {PHASH_HAMMING_THRESHOLD}/64)")
    p.add_argument("--quiet", "-q", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("serve", help="serve the swipe UI on the local network")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("train", help="fit the preference model on your swipes")
    p.add_argument("--l2", type=float, default=1.0, help="regularisation strength")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("stats", help="show library and model numbers")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("queue", help="print what the app would show next")
    p.add_argument("--mode", choices=list(queue_mod.MODES), default="learn")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_queue)

    p = sub.add_parser("apply", help="move dropped files to quarantine and file the keepers")
    p.add_argument("--confirm", action="store_true", help="actually move files (default: dry run)")
    p.add_argument("--no-sort", action="store_true", help="do not move keepers into year folders")
    p.add_argument("--no-quarantine", action="store_true", help="do not move dropped files")
    p.add_argument("--keep-duplicates", action="store_true",
                   help="leave byte-identical duplicates where they are")
    p.add_argument("--prune-empty", action="store_true",
                   help="remove source folders left empty by the move")
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser("undo", help="reverse an apply batch")
    p.add_argument("--batch", default=None, help="batch id (default: the most recent)")
    p.set_defaults(func=cmd_undo)

    p = sub.add_parser("empty-quarantine", help="permanently delete quarantined files")
    p.add_argument("--batch", default=None)
    p.add_argument("--confirm", action="store_true")
    p.set_defaults(func=cmd_empty)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (NotADirectoryError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
